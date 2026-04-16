'''
MuCo
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0
'''
"""
Merge LoRA adapter weights into the base model and save the full merged model.

Usage:
    python -m src.merge \
        --model_name Qwen/Qwen2-VL-2B-Instruct \
        --lora \
        --resume_from /path/to/lora-checkpoint \
        --output_dir ./merged_model \
        --add_emb_token True
"""
import torch
from transformers import AutoConfig
from src.arguments import parse_args
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.loader import EMB_TOKEN, MASK_TOKEN


def main():
    args = parse_args()
    args.model_backbone = 'qwen2_vl'

    assert args.lora, "--lora must be set to merge a LoRA checkpoint"
    assert args.resume_from, "--resume_from must point to a LoRA checkpoint directory"
    assert args.output_dir, "--output_dir must be set for saving the merged model"

    # Load processor
    processor = load_processor(args, args)
    special_tokens = [EMB_TOKEN, MASK_TOKEN]
    processor.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.special_tokens = special_tokens

    # Build model with LoRA structure
    model = MMEBModel.build(args, processor.tokenizer, special_tokens)

    # Load LoRA weights and merge into base model (merge_and_unload)
    print(f"Loading and merging LoRA from: {args.resume_from}")
    model.load(args.resume_from, args, is_trainable=False)
    model.eval()

    # Save merged model weights and processor
    print(f"Saving merged model to: {args.output_dir}")
    model.save(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Overwrite config.json with the original base model config
    # (save_pretrained may write model_type "qwen2_vl_text" from the inner language model)
    import json, os
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "r") as f:
        saved_config = json.load(f)

    def fix_model_type(obj):
        if isinstance(obj, dict):
            if "model_type" in obj:
                obj["model_type"] = "qwen2_vl"
            for v in obj.values():
                fix_model_type(v)
        elif isinstance(obj, list):
            for v in obj:
                fix_model_type(v)

    fix_model_type(saved_config)
    with open(config_path, "w") as f:
        json.dump(saved_config, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
