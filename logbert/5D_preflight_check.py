from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _3B_load_to_pytorch_logbert import MASK_TOKEN, PAD_TOKEN, build_loaders


DEFAULT_REPORTS_DIR = THIS_DIR / "reports"
DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight preflight checks for the LogBERT pipeline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-len-percentile", type=float, default=95.0)
    parser.add_argument("--max-len-cap", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--inspect-batches", type=int, default=4)
    return parser.parse_args()


def load_train_module():
    module_path = THIS_DIR / "4B_train_logbert.py"
    spec = importlib.util.spec_from_file_location("logbert_train_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sampled_token_stats(loader, unk_id: int, inspect_batches: int) -> dict[str, float]:
    total_tokens = 0
    total_nonpad = 0
    total_unk = 0
    batch_count = 0

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        total_tokens += int(input_ids.numel())
        total_nonpad += int(attention_mask.sum().item())
        total_unk += int((input_ids == unk_id).sum().item())
        batch_count += 1
        if batch_count >= inspect_batches:
            break

    return {
        "sampled_batches": int(batch_count),
        "sampled_tokens": int(total_tokens),
        "sampled_nonpad_tokens": int(total_nonpad),
        "sampled_unk_tokens": int(total_unk),
        "unk_rate_over_all_positions": float(total_unk / max(total_tokens, 1)),
        "unk_rate_over_nonpad_positions": float(total_unk / max(total_nonpad, 1)),
    }


def main() -> None:
    args = parse_args()
    train_module = load_train_module()

    train_loader, val_loader, test_loader, data_meta = build_loaders(
        batch_train=args.batch_size,
        batch_eval=args.batch_size,
        seed=0,
        root=args.data_root,
        max_len=args.max_len,
        max_len_percentile=args.max_len_percentile,
        max_len_cap=args.max_len_cap,
        min_freq=args.min_freq,
    )

    token_to_id = data_meta["token_to_id"]
    vocab_size = int(data_meta["vocab_size"])
    max_len = int(data_meta["max_len"])
    pad_id = int(token_to_id[PAD_TOKEN])
    mask_id = int(token_to_id[MASK_TOKEN])
    cls_id = int(token_to_id["[CLS]"])
    unk_id = int(token_to_id["[UNK]"])

    train_batch = next(iter(train_loader))
    input_ids = train_batch["input_ids"]
    attention_mask = train_batch["attention_mask"]

    masked_ids, mlm_labels = train_module.mlm_mask_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_id=pad_id,
        cls_id=cls_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        mask_prob=args.mask_prob,
    )

    model = train_module.LogBERTModel(
        vocab_size=vocab_size,
        max_len=max_len,
        pad_id=pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(masked_ids, attention_mask)
        mlm_logits = outputs["mlm_logits"]
        cls_output = outputs["cls_output"]
        loss = torch.nn.functional.cross_entropy(
            mlm_logits.reshape(-1, vocab_size),
            mlm_labels.reshape(-1),
            ignore_index=train_module.IGNORE_INDEX,
        )

    sequence_build_config = data_meta.get("sequence_build_config", {})
    variant_id = sequence_build_config.get("variant_id", Path(data_meta["root"]).name)
    report_path = DEFAULT_REPORTS_DIR / f"preflight_{variant_id}.json"

    report = {
        "data_root": data_meta["root"],
        "variant_id": variant_id,
        "sequence_build_config": sequence_build_config,
        "counts": {
            "n_train_total": int(data_meta["n_train_total"]),
            "n_train_benign": int(data_meta["n_train_benign"]),
            "n_train_attack": int(data_meta["n_train_attack"]),
            "n_val_total": int(data_meta["n_val_total"]),
            "n_val_benign": int(data_meta["n_val_benign"]),
            "n_val_attack": int(data_meta["n_val_attack"]),
            "n_test_total": int(data_meta["n_test_total"]),
            "n_test_benign": int(data_meta["n_test_benign"]),
            "n_test_attack": int(data_meta["n_test_attack"]),
        },
        "class_coverage": {
            "val_has_both_classes": bool(data_meta["n_val_benign"] > 0 and data_meta["n_val_attack"] > 0),
            "test_has_both_classes": bool(data_meta["n_test_benign"] > 0 and data_meta["n_test_attack"] > 0),
        },
        "vocab": {
            "vocab_size": vocab_size,
            "max_len": max_len,
            "vocab_train_scope": data_meta["vocab_train_scope"],
        },
        "batch_shapes": {
            "input_ids": list(input_ids.shape),
            "attention_mask": list(attention_mask.shape),
            "masked_ids": list(masked_ids.shape),
            "mlm_logits": list(mlm_logits.shape),
            "cls_output": list(cls_output.shape),
        },
        "forward_check": {
            "masked_token_count": int((mlm_labels != train_module.IGNORE_INDEX).sum().item()),
            "loss": float(loss.item()),
            "loss_is_finite": bool(torch.isfinite(loss).item()),
            "logits_finite": bool(torch.isfinite(mlm_logits).all().item()),
            "cls_finite": bool(torch.isfinite(cls_output).all().item()),
        },
        "sampled_token_stats": {
            "train": sampled_token_stats(train_loader, unk_id=unk_id, inspect_batches=args.inspect_batches),
            "val": sampled_token_stats(val_loader, unk_id=unk_id, inspect_batches=args.inspect_batches),
            "test": sampled_token_stats(test_loader, unk_id=unk_id, inspect_batches=args.inspect_batches),
        },
    }

    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved preflight report to: {report_path}")


if __name__ == "__main__":
    main()
