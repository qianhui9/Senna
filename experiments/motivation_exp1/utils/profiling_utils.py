"""Temporary transparent instrumentation; never retain features between requests."""
from contextlib import contextmanager, nullcontext
import functools
import hashlib
import inspect
import json
import time


@contextmanager
def replace_attribute(obj, name, replacement):
    """Restore inherited attributes as inherited, including on exceptions."""
    existed = name in vars(obj)
    saved = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, saved)
        else:
            delattr(obj, name)


def shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape(v) for v in value]
    return None


def tensor_digest(tensor):
    data = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


class CudaTimeline:
    """Single-device current-stream elapsed intervals, with no inner syncs."""
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.intervals = []

    def begin(self, name):
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        with self.torch.cuda.device(self.device):
            start.record(self.torch.cuda.current_stream(self.device))
        self.intervals.append((name, start, end))
        return end

    def finish(self, end):
        with self.torch.cuda.device(self.device):
            end.record(self.torch.cuda.current_stream(self.device))

    def totals(self):
        result = {}
        self.torch.cuda.synchronize(self.device)
        for name, start, end in self.intervals:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end)
        return result


class ModuleHooks:
    """Only hooks around actual modules; neither inputs nor outputs are returned."""
    def __init__(self, modules, timeline=None):
        self.modules, self.timeline = modules, timeline
        self.handles, self.calls, self.stacks = [], {}, {}

    def __enter__(self):
        for name, module in self.modules.items():
            self.calls[name], self.stacks[name] = [], []
            def before(module, inputs, name=name):
                end = self.timeline.begin(name) if self.timeline else None
                self.stacks[name].append(end)
                self.calls[name].append({"input_shape": shape(inputs), "output_shape": None})
            def after(module, inputs, output, name=name):
                end = self.stacks[name].pop()
                if self.timeline:
                    self.timeline.finish(end)
                self.calls[name][-1]["output_shape"] = shape(output)
            self.handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def cache_sequence_length(cache):
    """Inspect KV metadata only; support the installed legacy tuple and Cache API."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        key = cache[0][0]
        if key.ndim != 4:
            raise ValueError("Expected [batch, heads, sequence, head_dim] KV cache")
        return int(key.shape[-2])
    raise ValueError("Unsupported KV cache: cannot establish prefill/decode boundary")


def classify_llm_forward(arguments):
    """Classify the real forward by its input sequence and incoming KV cache."""
    embeds = arguments.get("inputs_embeds")
    inputs = embeds if embeds is not None else arguments.get("input_ids")
    if inputs is None:
        raise ValueError("LLM forward has neither inputs_embeds nor input_ids")
    tokens = int(inputs.shape[-2] if embeds is not None else inputs.shape[-1])
    cached = cache_sequence_length(arguments.get("past_key_values"))
    phase = "decode" if cached else "prefill"
    if tokens < 1 or (phase == "decode" and tokens != 1):
        raise ValueError("Expected nonempty prompt and single-token cached decode")
    return {"phase": phase, "input_shape": shape(inputs), "input_tokens": tokens,
            "input_kind": "inputs_embeds" if embeds is not None else "input_ids",
            "past_kv_tokens": cached}


class LLMForwardTiming:
    """Transparent forward wrapper; no model outputs or KV tensors are retained."""
    def __init__(self, model, timeline=None, record_ranges=False):
        self.model, self.timeline, self.record_ranges = model, timeline, record_ranges
        self.calls, self.interval_indices = [], []
        self.in_forward = False

    def __enter__(self):
        original = self.model.forward
        signature = inspect.signature(original)

        @functools.wraps(original)
        def forward(*a, **kw):
            call = classify_llm_forward(signature.bind(*a, **kw).arguments)
            if self.record_ranges:
                import torch
                scope = torch.profiler.record_function("senna.llm_" + call["phase"])
            else:
                scope = nullcontext()
            with scope:
                end = self.timeline.begin("llm_" + call["phase"]) if self.timeline else None
                index = len(self.timeline.intervals) - 1 if self.timeline else None
                self.in_forward = True
                try:
                    result = original(*a, **kw)
                finally:
                    if self.timeline:
                        self.timeline.finish(end)
                    self.in_forward = False
            call["present_kv_tokens"] = cache_sequence_length(result.past_key_values)
            self.calls.append(call)
            self.interval_indices.append(index)
            return result

        self.replacement = replace_attribute(self.model, "forward", forward)
        self.replacement.__enter__()
        return self

    def __exit__(self, *exc):
        return self.replacement.__exit__(*exc)

    def validate(self, output_tokens):
        """Greedy prefill emits token one; cached forwards emit the remaining tokens."""
        if not self.calls or self.calls[0]["phase"] != "prefill":
            raise ValueError("Generation did not start with prefill")
        if len(self.calls) != output_tokens:
            raise ValueError("Forward count differs from generated token count")
        previous = 0
        for i, call in enumerate(self.calls):
            if call["phase"] != ("prefill" if i == 0 else "decode"):
                raise ValueError("Expected exactly one prefill per generation request")
            if call["past_kv_tokens"] != previous:
                raise ValueError("KV cache does not continue from the preceding forward")
            previous += call["input_tokens"]
            if call["present_kv_tokens"] != previous:
                raise ValueError("KV cache growth differs from consumed input length")
            if self.timeline:
                _, start, end = self.timeline.intervals[self.interval_indices[i]]
                call["duration_ms"] = start.elapsed_time(end)
            else:
                call["duration_ms"] = None
        return {"llm_prefill_calls": 1, "llm_decode_calls": len(self.calls) - 1,
                "llm_prefill_input_tokens": self.calls[0]["input_tokens"],
                "llm_decode_input_tokens": sum(c["input_tokens"] for c in self.calls[1:]),
                "llm_forward_calls": json.dumps(self.calls), "llm_split_valid": True}


def original_request(args, tokenizer, model, processor, capture_input=False):
    """Original local helper, no timing hooks. Capture ids only for equivalence."""
    from data_tools.senna_qa_utils import eval_multi_img_model_wo_init
    generated, snapshot = {}, {}
    original = model.generate
    @functools.wraps(original)
    def wrapper(*a, **kw):
        if capture_input:
            snapshot.update(input_ids_sha256=tensor_digest(a[0]), images_sha256=tensor_digest(kw["images"]), image_shape=shape(kw["images"]), image_sizes=kw["image_sizes"], generation={k: kw[k] for k in ("do_sample", "temperature", "top_p", "num_beams", "max_new_tokens", "use_cache")})
        result = original(*a, **kw)
        generated["ids"] = result
        return result
    with replace_attribute(model, "generate", wrapper):
        answer = eval_multi_img_model_wo_init(args, tokenizer, model, processor)
    return answer, generated["ids"].detach().cpu().tolist(), snapshot


def timed_request(args, tokenizer, model, processor, capture_input=False, record_ranges=False):
    """Invoke the exact local evaluation helper under reversible wrappers/hooks.

    CUDA intervals include stream idle time/host dispatch gaps between events.
    Parents are decomposed, never added to their children. Cross-device models
    get synchronized request wall time and metadata only, not stage estimates.
    """
    from contextlib import ExitStack
    import torch
    from data_tools import senna_qa_utils as qa
    from .stats_utils import latency_partition, llm_latency_partition

    devices = sorted({str(p.device) for p in model.parameters()})
    single = len(devices) == 1 and devices[0].startswith("cuda")
    device = model.device
    timeline = CudaTimeline(torch, device) if single else None
    model_devices = [torch.device(d) for d in devices if d.startswith("cuda")]
    def synchronize():
        for d in model_devices:
            torch.cuda.synchronize(d)
    base = model.get_model()
    modules = {"vision_tower": model.get_vision_tower(), "projector": base.mm_projector, "img_adapter": base.img_adapter}
    times, state, snapshot = {}, {}, {}
    load_original, process_original, tokenize_original = qa.load_images, qa.process_images, qa.tokenizer_image_token
    generate_original, prepare_original = model.generate, model.prepare_inputs_labels_for_multimodal

    def load(*a, **kw):
        start = time.perf_counter()
        result = load_original(*a, **kw)
        times["image_file_loading_ms"] = (time.perf_counter() - start) * 1000
        state["original_image_sizes"] = [x.size for x in result]
        return result

    def process(*a, **kw):
        start = time.perf_counter()
        result = process_original(*a, **kw)
        times["image_preprocessing_ms"] = (time.perf_counter() - start) * 1000
        state["transfer_start"] = time.perf_counter()
        return result

    def tokenize(*a, **kw):
        synchronize()
        times["host_to_device_ms"] = (time.perf_counter() - state["transfer_start"]) * 1000
        result = tokenize_original(*a, **kw)
        state["transfer_start"] = time.perf_counter()
        return result

    def prepare(*a, **kw):
        # Senna.forward also calls this helper during cached decode, where it
        # immediately returns without images. That work belongs to Decode;
        # subtracting it as preparation would overlap the forward interval.
        if llm.in_forward:
            return prepare_original(*a, **kw)
        end = timeline.begin("preparation_inclusive") if timeline else None
        try:
            return prepare_original(*a, **kw)
        finally:
            if timeline:
                timeline.finish(end)

    def generate(*a, **kw):
        synchronize()
        times["host_to_device_ms"] += (time.perf_counter() - state["transfer_start"]) * 1000
        state["input_shape"] = shape(kw["images"])
        state["input_tokens"] = a[0].shape[-1]
        if capture_input:
            snapshot.update(input_ids_sha256=tensor_digest(a[0]), images_sha256=tensor_digest(kw["images"]), image_shape=shape(kw["images"]), image_sizes=kw["image_sizes"], generation={k: kw[k] for k in ("do_sample", "temperature", "top_p", "num_beams", "max_new_tokens", "use_cache")})
        for d in model_devices:
            torch.cuda.reset_peak_memory_stats(d)
        start = time.perf_counter()
        end = timeline.begin("generation_inclusive") if timeline else None
        result = generate_original(*a, **kw)
        if timeline:
            timeline.finish(end)
        synchronize()
        times["generation_wall_ms"] = (time.perf_counter() - start) * 1000
        state["generated_ids"] = result
        return result

    with ExitStack() as stack:
        hooks = stack.enter_context(ModuleHooks(modules, timeline))
        llm = stack.enter_context(LLMForwardTiming(model, timeline, record_ranges))
        for obj, name, wrapper in ((qa, "load_images", load), (qa, "process_images", process), (qa, "tokenizer_image_token", tokenize), (model, "generate", generate), (model, "prepare_inputs_labels_for_multimodal", prepare)):
            stack.enter_context(replace_attribute(obj, name, wrapper))
        synchronize()
        start = time.perf_counter()
        answer = qa.eval_multi_img_model_wo_init(args, tokenizer, model, processor)
        synchronize()
        times["end_to_end_ms"] = (time.perf_counter() - start) * 1000
    totals = timeline.totals() if timeline else {}
    returned_count = int(state["generated_ids"].shape[-1])
    # The inspected Senna.generate passes inputs_embeds (no input_ids) to
    # Transformers; its returned decoder sequence starts with one BOS token.
    bos = model.generation_config.bos_token_id
    has_bos = returned_count > 0 and state["generated_ids"][0, 0].item() == bos
    metadata = {"input_shape": json.dumps(state["input_shape"]), "original_image_sizes": json.dumps(state["original_image_sizes"]), "input_tokens": state["input_tokens"], "output_tokens": returned_count - 1 if has_bos else returned_count, "returned_output_token_count": returned_count, "output_bos_prefix_removed": has_bos,
                "vision_tower_calls": len(hooks.calls["vision_tower"]), "projector_calls": len(hooks.calls["projector"]), "img_adapter_calls": len(hooks.calls["img_adapter"]), "module_shapes": json.dumps(hooks.calls), "devices": json.dumps(devices), "timing_decomposition_valid": single,
                "peak_allocated_bytes": json.dumps({str(d): torch.cuda.max_memory_allocated(d) for d in model_devices}), "peak_reserved_bytes": json.dumps({str(d): torch.cuda.max_memory_reserved(d) for d in model_devices})}
    vision_shape = hooks.calls["vision_tower"][0]["output_shape"] if hooks.calls["vision_tower"] else None
    adapter_shapes = [c["output_shape"] for c in hooks.calls["img_adapter"]]
    metadata["vision_tokens_per_image"] = vision_shape[-2] if vision_shape else ""
    metadata["visual_tokens_total"] = sum(s[-2] * s[0] for s in adapter_shapes) if adapter_shapes else 0
    metadata.update(llm.validate(metadata["output_tokens"]))
    metadata["llm_split_valid"] = single
    if single:
        times.update(vision_tower_ms=totals.get("vision_tower", 0), projector_ms=totals.get("projector", 0), img_adapter_ms=totals.get("img_adapter", 0), preparation_inclusive_ms=totals.get("preparation_inclusive", 0), generate_inclusive_ms=totals["generation_inclusive"])
        try:
            times.update(latency_partition(times["end_to_end_ms"], times["image_file_loading_ms"], times["image_preprocessing_ms"], times["host_to_device_ms"], times["generate_inclusive_ms"], times["preparation_inclusive_ms"], times["vision_tower_ms"], times["projector_ms"], times["img_adapter_ms"]))
            times.update(llm_latency_partition(times["llm_generation_ms"], totals.get("llm_prefill", 0), totals.get("llm_decode", 0), times["end_to_end_ms"]))
        except ValueError as exc:
            metadata["timing_decomposition_valid"] = False
            metadata["llm_split_valid"] = False
            metadata["decomposition_error"] = str(exc)
    for name in ("vision_tower_ms", "projector_ms", "img_adapter_ms", "multimodal_preparation_ms", "llm_generation_ms", "other_ms", "visual_path_ms", "vision_latency_ratio", "projector_latency_ratio", "visual_path_latency_ratio", "llm_generation_latency_ratio", "ideal_visual_path_speedup_bound", "preparation_inclusive_ms", "generate_inclusive_ms"):
        times.setdefault(name, "")
    for name in ("llm_prefill_ms", "llm_decode_ms", "llm_generation_overhead_ms", "llm_prefill_latency_ratio", "llm_decode_latency_ratio", "llm_generation_overhead_latency_ratio", "llm_decode_ms_per_token"):
        times.setdefault(name, "")
    if metadata["llm_split_valid"] and metadata["llm_decode_calls"]:
        times["llm_decode_ms_per_token"] = times["llm_decode_ms"] / metadata["llm_decode_calls"]
    metadata.setdefault("decomposition_error", "")
    return answer, state["generated_ids"].detach().cpu().tolist(), snapshot, {**times, **metadata}
