"""Temporary transparent instrumentation; never retain features between requests."""
from contextlib import contextmanager
import functools
import hashlib
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


def timed_request(args, tokenizer, model, processor, capture_input=False):
    """Invoke the exact local evaluation helper under reversible wrappers/hooks.

    CUDA intervals include stream idle time/host dispatch gaps between events.
    Parents are decomposed, never added to their children. Cross-device models
    get synchronized request wall time and metadata only, not stage estimates.
    """
    from contextlib import ExitStack
    import torch
    from data_tools import senna_qa_utils as qa
    from .stats_utils import latency_partition

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
    if single:
        times.update(vision_tower_ms=totals.get("vision_tower", 0), projector_ms=totals.get("projector", 0), img_adapter_ms=totals.get("img_adapter", 0), preparation_inclusive_ms=totals.get("preparation_inclusive", 0), generate_inclusive_ms=totals["generation_inclusive"])
        try:
            times.update(latency_partition(times["end_to_end_ms"], times["image_file_loading_ms"], times["image_preprocessing_ms"], times["host_to_device_ms"], times["generate_inclusive_ms"], times["preparation_inclusive_ms"], times["vision_tower_ms"], times["projector_ms"], times["img_adapter_ms"]))
        except ValueError as exc:
            metadata["timing_decomposition_valid"] = False
            metadata["decomposition_error"] = str(exc)
    for name in ("vision_tower_ms", "projector_ms", "img_adapter_ms", "multimodal_preparation_ms", "llm_generation_ms", "other_ms", "visual_path_ms", "vision_latency_ratio", "projector_latency_ratio", "visual_path_latency_ratio", "llm_generation_latency_ratio", "ideal_visual_path_speedup_bound", "preparation_inclusive_ms", "generate_inclusive_ms"):
        times.setdefault(name, "")
    metadata.setdefault("decomposition_error", "")
    return answer, state["generated_ids"].detach().cpu().tolist(), snapshot, {**times, **metadata}
