"""Run Senna on aligned six-camera inputs of one nuScenes sample."""

import argparse
import json
import os
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from data_tools.senna_inputs import (
    CAMERAS, IMAGE_PROMPT, PROTOCOL_VERSION, check_model, current_speed,
    future_positions, navigation_command, parse_plan, planning_question, sample_images,
)
from data_tools.senna_qa_utils import eval_multi_img_model_wo_init
from llava.model.builder import load_senna_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("SENNA_MODEL_PATH", "/data/qh/phdCode/work4/models/Senna"))
    parser.add_argument("--lora-path", help="Fine-tuned LoRA checkpoint; keep --model-path set to the base weights")
    parser.add_argument("--dataroot", default=os.environ.get("NUSCENES_ROOT", "/data/qh/nuscenes/v1.0-trainval"))
    parser.add_argument("--version", default="v1.0-trainval")
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--sample-token")
    sample.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--speed", type=float, help="Current speed in m/s; inferred from CAN bus if omitted")
    parser.add_argument("--navigation", choices=("go straight", "turn left", "turn right", "go straight and turn left", "go straight and turn right"), help="Route instruction; otherwise use offline future route")
    parser.add_argument("--question", help="Custom question; six-view image prompt is prepended")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--save-path")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_driving_context(nusc, dataroot, sample):
    return (
        current_speed(nusc, NuScenesCanBus(dataroot=dataroot), sample),
        navigation_command(future_positions(nusc, sample, steps=12)),
    )


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    record = nusc.get("sample", args.sample_token) if args.sample_token else nusc.sample[args.sample_index]
    images = sample_images(nusc, record)
    speed = args.speed if args.speed is not None else current_speed(nusc, NuScenesCanBus(dataroot=args.dataroot), record)
    navigation = args.navigation or navigation_command(future_positions(nusc, record, steps=12))
    question = IMAGE_PROMPT + (args.question or planning_question(speed, navigation))
    result = {
        "sample_token": record["token"], "speed_mps": speed, "navigation": navigation,
        "images": images, "question": question,
        "input_protocol": {"version": PROTOCOL_VERSION, "camera_order": list(CAMERAS),
                           "navigation_source": "user_route" if args.navigation else "offline_future_route_6s"},
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    check_model(args.model_path)
    tokenizer, model, image_processor, _ = load_senna_pretrained_model(
        args.model_path, None, model_name="llava", device_map=args.gpu
    )
    if args.lora_path:
        from train_tools.lora_io import load_lora_for_inference
        model = load_lora_for_inference(model, args.lora_path, args.model_path)
        result["lora_path"] = str(Path(args.lora_path).resolve())
    model.eval()
    inference_args = argparse.Namespace(
        model_path=args.model_path, model_base=None, query=question, conv_mode="llava_v1",
        image_file=images, sep=",", temperature=0, top_p=None, num_beams=1,
        max_new_tokens=args.max_new_tokens,
    )
    result["answer"] = eval_multi_img_model_wo_init(inference_args, tokenizer, model, image_processor)
    if not args.question:
        speed_plan, path_plan = parse_plan(result["answer"])
        result["parsed_plan"] = {"speed": speed_plan, "path": path_plan, "valid": speed_plan is not None}
    print(f"Sample: {record['token']}\nNavigation: {navigation}\nAnswer: {result['answer']}")
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
