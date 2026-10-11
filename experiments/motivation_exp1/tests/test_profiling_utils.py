import unittest
import torch

from experiments.motivation_exp1.utils.profiling_utils import ModuleHooks, replace_attribute
from experiments.motivation_exp1.utils.stats_utils import latency_partition


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


if __name__ == "__main__":
    unittest.main()
