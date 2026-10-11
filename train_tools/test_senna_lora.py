"""CPU regression tests for planning supervision, leakage, and visual checkpoint IO."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoTokenizer

from train_tools.senna_lora import encode_planning, load_records, PlanningCollator
from train_tools.lora_io import RESUME_FILES, latest_complete_checkpoint, load_visual_weights, mark_complete, save_visual_weights
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.sample = json.loads((root / "data/senna_nusc_val_plan_corrected.json").read_text())[0]
        cls.tokenizer = AutoTokenizer.from_pretrained(str(root.parent / "models/Senna"), use_fast=False, local_files_only=True)

    def test_only_answer_and_eos_supervised(self):
        ids, labels, length = encode_planning(self.sample, self.tokenizer, 1536)
        self.assertEqual(int((ids == IMAGE_TOKEN_INDEX).sum()), 6)
        self.assertTrue((labels[ids == IMAGE_TOKEN_INDEX] == IGNORE_INDEX).all())
        self.assertEqual(length, len(ids) + 762)
        answer = self.tokenizer.decode(labels[labels != IGNORE_INDEX]).strip()
        self.assertEqual(answer, self.sample["conversations"][1]["value"].strip() + "</s>")
        with self.assertRaisesRegex(ValueError, "expanded length"):
            encode_planning(self.sample, self.tokenizer, 800)

    def test_token_and_scene_leakage(self):
        with tempfile.TemporaryDirectory(prefix="senna_lora_tests_") as directory:
            train, val = Path(directory) / "train.json", Path(directory) / "val.json"
            train.write_text(json.dumps([self.sample]))
            val.write_text(json.dumps([self.sample]))
            with self.assertRaisesRegex(ValueError, "sample leakage"):
                load_records(train, val)
            other = copy.deepcopy(self.sample)
            other["token"] = "different_token"
            val.write_text(json.dumps([other]))
            with self.assertRaisesRegex(ValueError, "scene leakage"):
                load_records(train, val)
            other["scene_token"] = "different_scene"
            val.write_text(json.dumps([other]))
            self.assertEqual(len(load_records(train, val)), 1)

    def test_batch_padding_does_not_supervise_padding(self):
        rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]), "images": torch.zeros(6, 3, 2, 2)},
                {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2]), "images": torch.zeros(6, 3, 2, 2)}]
        batch = PlanningCollator(0)(rows)
        self.assertEqual(batch["images"].shape, (2, 6, 3, 2, 2))
        self.assertEqual(batch["labels"][1, 2].item(), IGNORE_INDEX)
        self.assertFalse(batch["attention_mask"][1, 2].item())

    def test_extra_visual_parameters_roundtrip(self):
        base = torch.nn.Module()
        base.model = torch.nn.Module()
        base.model.mm_projector = torch.nn.Linear(2, 2)
        base.model.img_adapter = torch.nn.Linear(2, 2)
        wrapper = SimpleNamespace(get_base_model=lambda: base)
        with tempfile.TemporaryDirectory(prefix="senna_lora_tests_") as directory:
            wanted = {k: v.clone() for k, v in base.state_dict().items()}
            save_visual_weights(wrapper, directory)
            with torch.no_grad():
                for p in base.parameters():
                    p.zero_()
            load_visual_weights(wrapper, directory)
            for k, v in base.state_dict().items():
                self.assertTrue(torch.equal(wanted[k], v))

    def test_resume_skips_interrupted_or_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="senna_lora_tests_") as directory:
            older = Path(directory) / "checkpoint-1"
            older.mkdir()
            for name in RESUME_FILES:
                (older / name).write_bytes(b"test checkpoint")
            mark_complete(older)
            newer = Path(directory) / "checkpoint-2"
            newer.mkdir()
            self.assertEqual(latest_complete_checkpoint(directory), str(older))
            for name in RESUME_FILES:
                (newer / name).write_bytes(b"test checkpoint")
            mark_complete(newer)
            self.assertEqual(latest_complete_checkpoint(directory), str(newer))
            # Corrupt contents without changing size, exercising the SHA256 guard.
            (newer / "optimizer.pt").write_bytes(b"bad! checkpoint")
            self.assertEqual(latest_complete_checkpoint(directory), str(older))


if __name__ == "__main__":
    unittest.main()
