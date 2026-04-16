'''
MuCo
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0
'''
import argparse
from typing import List, Literal

def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation Arguments")

    # --- Model Arguments ---
    model_args_group = parser.add_argument_group("Model Arguments")
    model_args_group.add_argument("--model_name", type=str, required=True, help="huggingface model name or path")
    model_args_group.add_argument("--model_type", type=str, default=None,
                                  help="model type, typically includes in config file, but sometimes needs manually add")
    model_args_group.add_argument("--processor_name", type=str, default=None,
                                  help="processor_name, huggingface model name or path")
    model_args_group.add_argument("--model_backbone", type=str, default=None, help="HF model type")
    model_args_group.add_argument("--checkpoint_path", type=str, default=None,
                                  help="a local model path, could be a LoRA version")
    model_args_group.add_argument("--pooling", type=str, default='last', help="pooling method for encoder")
    model_args_group.add_argument("--normalize", action="store_true",
                                  help="normalize query and passage representations")
    model_args_group.add_argument("--temperature", type=float, default=0.02, help="temperature for softmax")
    model_args_group.add_argument("--margin", type=float, default=0.0, help="margin for softmax")
    model_args_group.add_argument("--lora", action="store_true", help="do parameter-efficient fine-tuning with lora")
    model_args_group.add_argument("--lora_r", type=int, default=16, help="lora r")
    model_args_group.add_argument("--lora_alpha", type=int, default=64, help="lora alpha")
    model_args_group.add_argument("--lora_dropout", type=float, default=0.1, help="lora dropout")
    model_args_group.add_argument("--lora_target_modules", type=str,
                                  default="qkv_proj,o_proj,gate_proj,up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
                                  help="lora target modules")
    model_args_group.add_argument("--num_crops", type=int, default=16, help="number of crops used in image encoder")
    model_args_group.add_argument("--add_emb_token", type=bool, default=True, help="add an embed token")
    model_args_group.add_argument("--use_gen_loss", action="store_true", help="use a generative loss")
    model_args_group.add_argument("--only_gen_loss", action="store_true", help="use only a generative loss")
    model_args_group.add_argument("--use_predictor", action="store_true", help="use a predictor")
    model_args_group.add_argument("--device", type=str, default="cuda",
                                  help="use cuda for single GPU inference, if multiple GPUs are available it will use DP automatically")
    model_args_group.add_argument("--use_bidirectional", action="store_true", help="use a bidirectional attention")
    model_args_group.add_argument("--mean_pooling", action="store_true", help="use the mean pooling method")

    # --- Data Arguments ---
    data_args_group = parser.add_argument_group("Data Arguments")
    data_args_group.add_argument("--dataset_config", type=str, default='./eval_configs/image.yaml',
                                 help="yaml file with dataset configuration")
    data_args_group.add_argument("--wds_train_dir", type=str, default=None,
                                 help="Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path")
    data_args_group.add_argument("--dataset_name", type=str, default=None, help="huggingface dataset name")
    data_args_group.add_argument("--subset_name", type=List[str], default=None, help="Useful for datasets with subsets")
    data_args_group.add_argument("--dataset_split", type=str, default='train', help="dataset split")
    data_args_group.add_argument("--num_sample_per_subset", type=int, default=None,
                                 help="number of training samples per subset")
    data_args_group.add_argument("--data_basedir", type=str, default=None, help="tmp, will be deprecated")
    data_args_group.add_argument("--image_dir", type=str, default=None, help="Image directory path")
    data_args_group.add_argument("--encode_output_path", type=str, default=None, help="encode output path")
    data_args_group.add_argument("--max_len", type=int, default=None,
                                 help="The maximum total input sequence length after tokenization. Use with caution, since it may truncate text prompts due to large image lengths.")
    data_args_group.add_argument("--embedding_type", type=str, default="", help="embedding type")
    data_args_group.add_argument("--image_resolution", type=str, default=None,
                                 help="for models i.e. LLaVA-next and Qwen, resize images first, none means using original image resolution. This is only works when `--resize_use_processor false`.")
    data_args_group.add_argument("--resize_use_processor", action="store_true", default=True,
                                 help="Resize visual inputs insides processor, e.g. Qwen2VLImageProcessor, instead of by our code.")
    data_args_group.add_argument("--resize_min_pixels", type=int, default=28*28*256,
                                 help="The min pixels of the image to resize the image. This is only works when `--resize_use_processor true`.")
    data_args_group.add_argument("--resize_max_pixels", type=int, default=28*28*1280,
                                 help="The max pixels of the image to resize the image. This is only works when `--resize_use_processor true`.")
    data_args_group.add_argument("--image_decay_factor", type=float, default=None,
                                 help="The image decay factor for resizing temporal images")
    data_args_group.add_argument("--num_hardneg", type=int, default=0, help="hard negative number")
    data_args_group.add_argument("--use_torch_loader", action="store_true", help="use a torch loader")
    data_args_group.add_argument("--meta_folder", type=str, default=None, help="meta folder")
    data_args_group.add_argument("--image_folder", type=str, default=None, help="image folder")
    data_args_group.add_argument("--n_samples", type=int, default=None, help='Num of samples per dataset')

    # --- Training Arguments ---
    training_args_group = parser.add_argument_group("Training Arguments")
    training_args_group.add_argument("--output_dir", type=str, default=None,
                                     help="directory for saving trained models")
    training_args_group.add_argument("--resume_from", type=str, default=None,
                                     help="`auto` will detect if any previous checkpoints should be resumed. or specify specific step of the checkpoint.")
    training_args_group.add_argument("--hf_cache_dir", type=str, default=None, help="folder path to save any hf models")
    training_args_group.add_argument("--project_name", type=str, default=None, help="project name")
    training_args_group.add_argument("--logging_steps", type=int, default=1, help="logging steps")
    training_args_group.add_argument("--num_train_epochs", type=int, default=1, help="number of training epochs")
    training_args_group.add_argument("--accum_freq", type=int, default=1)
    training_args_group.add_argument("--mixed_precision", type=str, choices=['no', 'fp16', 'bf16'], default='no')
    training_args_group.add_argument("--dataloader_num_workers", type=int, default=0,
                                     help="Number of data loader workers.")
    training_args_group.add_argument("--deepspeed", type=str, default='zero2', help="Path to DeepSpeed config file.")
    training_args_group.add_argument("--per_device_train_batch_size", type=int, default=8,
                                     help="Batch size per device during training.")
    training_args_group.add_argument("--per_device_eval_batch_size", type=int, default=8,
                                     help="Batch size per device during evaluation.")
    training_args_group.add_argument("--gradient_checkpointing", action="store_true",
                                     help="Whether to use gradient checkpointing.")
    training_args_group.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate.")
    training_args_group.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of training steps.")
    training_args_group.add_argument("--warmup_steps", type=int, default=100,
                                     help="Number of steps for the warmup phase.")
    training_args_group.add_argument("--checkpointing_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    training_args_group.add_argument("--use_8bit_adamw", action="store_true", help="use the 8bit adamw")
    training_args_group.add_argument("--pretraining", action="store_true", help="pretraining")

    return parser.parse_args()

