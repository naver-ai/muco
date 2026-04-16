'''
Derived from VLM2Vec (https://github.com/TIGER-AI-Lab/VLM2Vec)
MIT License
but heavily modified
'''
import datetime
import logging
import json
import random
import time

import numpy as np
import os
import pickle
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from typing import Any
import re

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from datasets import Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

from src.arguments import parse_args
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from src.eval_utils.metrics import RankingMetrics
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils import batch_to_device, print_rank, print_master
from src.loader import EMB_TOKEN, MASK_TOKEN, VLM_IMAGE_TOKENS, TEMPLATE_BASE
import multiprocessing
from multiprocessing import Pool, cpu_count
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


def encode_embeddings(
    processor, # only for debugging
    model: MMEBModel,
    loader: DataLoader,
    args: Any,
    full_dataset: Dataset,
    encode_side: str,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_embeds = []
    local_gt_infos = []
    local_max_len = 0

    model.eval()
    with torch.no_grad():
        for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank {local_rank})", disable=local_rank > 0):
            texts = inputs['texts']
            new_texts = []
            for text, image, meta in zip(texts, inputs['images'], dataset_info):
                if len(text) > 0 and text[-1] != '\n':
                    text += '\n'
                text = re.sub(r'\n+', '\n', text)

                text = text.replace('<|image_1|>', VLM_IMAGE_TOKENS[args.model_backbone])
                text = text.replace('<|image_pad|>', VLM_IMAGE_TOKENS[args.model_backbone])
                if image is None:
                    text = text.replace(VLM_IMAGE_TOKENS[args.model_backbone], '')
                text = TEMPLATE_BASE.format(user_sentence=text)
                new_texts.append(text)
            texts = new_texts
            images = inputs['images']

            images = [image for image in images if image is not None]
            if len(images) == 0:
                images = None

            batch_inputs = processor(text=texts,
                                     images=images,
                                     padding=True,
                                     max_length=None,
                                     truncation=True,
                                     return_tensors='pt',
                                     )
            inputs = batch_inputs
            inputs = batch_to_device(inputs, args.device)

            with torch.no_grad() and torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Determine if encoding query or target based on available keys
                if encode_side == "qry":
                    output = model(qry=inputs)
                    reps = output["qry_reps"].detach()
                    local_gt_infos.extend(dataset_info)  # to retain all information per query
                else:
                    output = model(tgt=inputs)
                    reps = output["tgt_reps"].detach()
                    local_gt_infos.extend([info["cand_name"] for info in dataset_info])  # to retain ground-truth labels

            local_embeds.append(reps)

    if not local_embeds:
        # Handle cases where a rank gets no data
        return np.array([]), []

    embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()


    # === Gather embeddings and keys from all ranks ===
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        print_master(f"Gathering {encode_side} embeddings across all ranks...")

        # Use the more efficient all_gather_into_tensor for tensors
        output_shape = list(embeds_tensor.shape)
        output_shape[0] = full_dataset.num_rows
        embeds_tensor = embeds_tensor.to(args.device)
        gathered_embeds_tensor = torch.empty(output_shape, dtype=embeds_tensor.dtype, device=args.device)
        dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)
        final_embeddings = gathered_embeds_tensor.cpu().float().numpy()
        # Gather metadata, for which all_gather_object is appropriate
        gathered_gt_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt_infos, local_gt_infos)
        all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]
    else:
        all_gt_infos = local_gt_infos
        final_embeddings = embeds_tensor.cpu().float().numpy()

    return final_embeddings, all_gt_infos


def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # DEBUG PRINTS for Distributed Setup
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")
        torch.cuda.set_device(dist.get_rank())

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)

    args = parse_args()
    
    if args.encode_output_path is None:
        args.encode_output_path = os.path.join('/'.join(args.resume_from.split('/')[:-1]), f'eval_{os.path.basename(args.resume_from)}')
    os.makedirs(args.encode_output_path, exist_ok=True)

    # --- Model Loading ---
    args.model_backbone = 'qwen2_vl'

    processor = load_processor(args, args)

    if args.add_emb_token:
        special_tokens = [EMB_TOKEN, MASK_TOKEN]
        processor.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    else:
        special_tokens = None

    args.special_tokens = special_tokens

    model = MMEBModel.build(args, processor.tokenizer, special_tokens)
    model_backbone = get_backbone_name(hf_config=model.config)
    print_rank(f'Model_backbone: {model_backbone}')
    args.model_backbone = model_backbone

    if args.resume_from:
        checkpoint_path = args.resume_from
        print_rank(f"loaded model from {args.resume_from}")
        model.load(checkpoint_path, args, is_trainable=False)
    model.eval()

    args.device = f"cuda:{dist.get_rank()}"
    model = model.to(args.device)#, dtype=torch.bfloat16)
    with open(args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # 0. load dataset
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)
    
        if do_query or do_cand:
            if args.data_basedir is not None:
                # Construct full paths for data files if --data_basedir is provided
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if args.data_basedir and task_config.get(key):
                        task_config[key] = os.path.join(args.data_basedir, task_config[key])

            full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(model_args=args, data_args=args, **task_config)
            full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
            eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

            # Pad datasets to be divisible by world_size before splitting
            if dist.is_initialized():
                padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=local_rank, world_size=world_size)
                eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=local_rank, world_size=world_size)
            else:
                padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

        # --- 1. Compute Query Embeddings ---
        if do_query:
            print_master("Encoding queries...")
            eval_qry_collator = MultimodalEvalDataCollator(processor, args, "qry")
            eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=eval_qry_collator, num_workers=args.dataloader_num_workers)
            query_embeds, gt_infos = encode_embeddings(processor, model, eval_qry_loader, args, padded_qry_dataset, encode_side="qry", description=f"Queries for {dataset_name}")
            query_embeds = query_embeds[:len(full_eval_qry_dataset)]  # world_size>1, trim the padded data points
            gt_infos = gt_infos[:len(full_eval_qry_dataset)]
            if local_rank == 0:
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)
                with open(dataset_info_path, 'w') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info) + '\n')
                print_master(f"Saved query embeddings to {query_embed_path}")
            if dist.is_initialized():
                dist.barrier()

        # --- 2. Compute Candidate Embeddings ---
        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_collator = MultimodalEvalDataCollator(processor, args, "cand")
            eval_cand_loader = DataLoader(eval_cand_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=eval_cand_collator, num_workers=args.dataloader_num_workers)

            cand_embeds, all_cand_ids = encode_embeddings(processor, model, eval_cand_loader, args, padded_cand_dataset, encode_side="cand", description=f"Candidates for {dataset_name}")
            cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]  # world_size>1, trim the padded data points
            all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]

            if local_rank == 0:
                cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}
                with open(cand_embed_path, 'wb') as f: pickle.dump(cand_embed_dict, f)
                print_master(f"Saved candidate embeddings to {cand_embed_path}")

        if dist.is_initialized():
            dist.barrier()

        # --- 3. Compute Scores (on master rank only) ---
        if local_rank == 0:
            score_path = os.path.join(args.encode_output_path, f"{dataset_name}_score.json")
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                    print_master(formatted)
                    continue
                except Exception as e:
                    print_master(f"Failed to load score for {dataset_name}, skipping {dataset_name}")
            with open(query_embed_path, 'rb') as f: qry_embeds = pickle.load(f)
            with open(cand_embed_path, 'rb') as f: cand_embed_dict = pickle.load(f)
            gt_infos = [json.loads(l) for l in open(dataset_info_path)]
            pred_dicts = []

            rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
            if rank_against_all_candidates:
                cand_keys = list(cand_embed_dict.keys())
                cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
                cosine_scores = np.dot(qry_embeds, cand_embeds.T)
                ranked_candids = np.argsort(-cosine_scores, axis=1)
                for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None
                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [cand_keys[i] for i in ranked_candid],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })
            else:
                for qid, (qry_embed, gt_info) in tqdm(enumerate(zip(qry_embeds, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    cand_embeds = np.stack([cand_embed_dict[key] for key in gt_info["cand_names"]])
                    cosine_score = np.dot(qry_embed, cand_embeds.T)
                    ranked_candids = np.argsort(-cosine_score)
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [gt_info["cand_names"][i] for i in ranked_candids],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })

            score_path = os.path.join(args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(args.encode_output_path, f"{dataset_name}_pred.jsonl")

            metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
            metrics = RankingMetrics(metrics_to_report)
            score_dict = metrics.evaluate(pred_dicts)
            formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
            score_dict["num_pred"] = len(pred_dicts)
            score_dict["num_data"] = len(gt_infos)
            print_master(f"Score of {dataset_name}:")
            print_master(formatted)
            print_master(f"Outputting final score to: {score_path}")
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)
            with open(pred_path, "w") as f:
                for pred in pred_dicts:
                    f.write(json.dumps(pred) + '\n')


if __name__ == "__main__":
    main()
