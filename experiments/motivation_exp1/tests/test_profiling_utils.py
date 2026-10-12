import unittest
import torch

from experiments.motivation_exp1.utils.profiling_utils import LLMForwardTiming, ModuleHooks, cache_sequence_length, classify_llm_forward, replace_attribute
from experiments.motivation_exp1.utils.stats_utils import latency_partition, llm_latency_partition


class ProfilingTests(unittest.TestCase):
    def test_disjoint_timing_partition(self):
        p = latency_partition(100, 5, 10, 2, 80, 30, 20, 5, 1)
        self.assertEqual(p["multimodal_preparation_ms"], 4)
        self.assertEqual(p["llm_generation_ms"], 50)
        self.assertEqual(p["other_ms"], 3)
        self.assertAlmostEqual(5 + 10 + 2 + 20 + 5 + 1 + p["multimodal_preparation_ms"] + p["llm_generation_ms"] + p["other_ms"], 100)
        self.assertAlmostEqual(p["ideal_visual_path_speedup_bound"], 100 / 74)

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            latency_partition(100, 5, 10, 2, 80, 10, 20, 5, 1)
        with self.assertRaises(ValueError):
            latency_partition(10, 5, 10, 2, 80, 30, 20, 5, 1)

    def test_inherited_method_restored_on_exception(self):
        class Parent:
            def f(self):
                return 1
        obj = Parent()
        with self.assertRaises(RuntimeError):
            with replace_attribute(obj, "f", lambda: 2):
                self.assertEqual(obj.f(), 2)
                raise RuntimeError()
        self.assertEqual(obj.f(), 1)
        self.assertNotIn("f", vars(obj))

    def test_existing_attribute_restored(self):
        class Obj:
            pass
        obj = Obj()
        obj.value = 3
        with replace_attribute(obj, "value", 4):
            self.assertEqual(obj.value, 4)
        self.assertEqual(obj.value, 3)

    def test_hooks_preserve_output_and_remove_handles(self):
        model = torch.nn.Linear(3, 4).eval()
        x = torch.ones(2, 3)
        plain = model(x)
        with ModuleHooks({"module": model}) as hooks:
            measured = model(x)
            self.assertTrue(torch.equal(plain, measured))
            self.assertEqual(hooks.calls["module"][0]["output_shape"], [2, 4])
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)

    def test_hooks_removed_after_failure(self):
        model = torch.nn.Linear(3, 4)
        with self.assertRaises(RuntimeError):
            with ModuleHooks({"module": model}):
                raise RuntimeError()
        self.assertFalse(model._forward_hooks)

    def test_llm_partition_keeps_control_overhead_and_closes(self):
        p = llm_latency_partition(50, 20, 28, 100)
        self.assertEqual(p["llm_generation_overhead_ms"], 2)
        self.assertEqual(sum(p[k] for k in ("llm_prefill_ms", "llm_decode_ms", "llm_generation_overhead_ms")), 50)
        self.assertEqual(p["llm_prefill_latency_ratio"], .2)
        self.assertEqual(llm_latency_partition(20, 18, 0, 100)["llm_decode_ms"], 0)

    def test_llm_partition_rejects_overlap_or_invalid_costs(self):
        for values in ((10, 8, 3, 100), (10, float("nan"), 0, 100), (10, 8, -1, 100), (10, 8, 1, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                llm_latency_partition(*values)

    def test_phase_follows_cache_not_input_length_alone(self):
        empty = ((torch.zeros(1, 2, 0, 4), torch.zeros(1, 2, 0, 4)),)
        cached = ((torch.zeros(1, 2, 9, 4), torch.zeros(1, 2, 9, 4)),)
        for cache in (None, (), empty):
            call = classify_llm_forward({"input_ids": torch.ones(1, 1), "past_key_values": cache})
            self.assertEqual(call["phase"], "prefill")
        call = classify_llm_forward({"input_ids": torch.ones(1, 1), "past_key_values": cached})
        self.assertEqual((call["phase"], call["past_kv_tokens"]), ("decode", 9))
        with self.assertRaises(ValueError):
            classify_llm_forward({"input_ids": torch.ones(1, 2), "past_key_values": cached})
        with self.assertRaises(ValueError):
            cache_sequence_length(object())

    def test_real_transformers_embedded_generation_preserves_tokens_and_signature(self):
        import inspect
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             bos_token_id=1, eos_token_id=None, pad_token_id=0)
        model = LlamaForCausalLM(config).eval()
        signature = inspect.signature(model.forward)
        with torch.inference_mode():
            embeds = model.get_input_embeddings()(torch.tensor([[1, 3, 4, 5, 6]]))
            for tokens in (1, 4):
                plain = model.generate(inputs_embeds=embeds, max_new_tokens=tokens, do_sample=False, use_cache=True)
                with LLMForwardTiming(model) as timing:
                    self.assertEqual(inspect.signature(model.forward), signature)
                    measured = model.generate(inputs_embeds=embeds, max_new_tokens=tokens, do_sample=False, use_cache=True)
                self.assertTrue(torch.equal(plain, measured))
                metrics = timing.validate(tokens)
                self.assertEqual(metrics["llm_decode_calls"], tokens - 1)
                self.assertEqual(metrics["llm_prefill_input_tokens"], 5)
                self.assertNotIn("forward", vars(model))
                self.assertFalse(timing.in_forward)
                with self.assertRaises(ValueError):
                    timing.validate(tokens + 1)

    def test_forward_wrapper_restored_after_model_failure(self):
        class Broken:
            def forward(self, input_ids=None, past_key_values=None):
                raise RuntimeError("failed forward")
        model = Broken()
        with self.assertRaises(RuntimeError):
            with LLMForwardTiming(model) as timing:
                model.forward(input_ids=torch.ones(1, 3))
        self.assertNotIn("forward", vars(model))
        self.assertFalse(timing.in_forward)


if __name__ == "__main__":
    unittest.main()
