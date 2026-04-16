'''
MuCo
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0
'''
import logging
import os
import glob
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import sys
import torch
import yaml
from transformers import HfArgumentParser
from src.arguments import parse_args
from src.loader import build_torch_dataloader
from src.model.model import MMEBModel
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger

from transformers import get_constant_schedule_with_warmup
import deepspeed

from peft import get_peft_model_state_dict

from tqdm import tqdm
import time

def to_device(batch, device):
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    return batch

def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)

    args = parse_args()
    args.report_to = []

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed)
    deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size * args.accum_freq

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=['tensorboard'],
        project_dir=args.output_dir,
        deepspeed_plugin=deepspeed_plugin,
    )


    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("logs")

    train_loader, tokenizer, added_new_tokens = build_torch_dataloader(args.meta_folder,
                                                                       args.image_folder,
                                                                       args.n_samples,
                                                                       batch_size=args.per_device_train_batch_size,
                                                                       add_emb_token=args.add_emb_token,
                                                                       model_name=args.model_name,
                                                                       hf_cache_dir=args.hf_cache_dir,
                                                                       pretraining=args.pretraining,
                                                                       )


    model = MMEBModel.build(args, tokenizer, added_new_tokens)
    model_backbone = get_backbone_name(hf_config=model.config)


    print_rank(f'model_backbone: {model_backbone}')

    global_step = 0
    n_epoch = 1

    if args.resume_from or (os.path.exists(args.output_dir) and len(glob.glob(os.path.join(args.output_dir, 'checkpoint-*')))):
        if args.resume_from:
            print('RESUME FROM')
            checkpoint_path = args.resume_from
        else:
            checkpoint_path = [checkpoint_folder for checkpoint_folder in sorted(glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))) if 'epoch' not in checkpoint_folder][-1]
            global_step = int(os.path.basename(checkpoint_path).split('-')[-1])
            # check n_epoch
            epoch_checkpoint_path_list = [checkpoint_folder for checkpoint_folder in sorted(glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))) if 'epoch' in checkpoint_folder]
            if len(epoch_checkpoint_path_list) > 0:
                n_epoch = int(epoch_checkpoint_path_list[-1].split('-')[-1]) + 1
        model.load(checkpoint_path, args)


    if args.use_8bit_adamw:
        import bitsandbytes as bnb   
        optimizer_fn = bnb.optim.AdamW8bit
    else:
        optimizer_fn = torch.optim.AdamW

    optimizer = optimizer_fn(model.encoder.parameters(),
                             lr=args.learning_rate,
                             )

    lr_scheduler = get_constant_schedule_with_warmup(
                                  optimizer=optimizer,
                                  num_warmup_steps=args.warmup_steps,
    )

    model.gradient_checkpointing_enable()


    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )


    model.train()

    train_loss = 0.0
    
    progress_bar = tqdm(range(global_step, args.max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.accum_freq > 1:
        accum_inputs, accum_features = [], {'query': [], 'query2': [], 'pos': [], 'pos2': [], 'neg': []}

    batch_idx = -1
    while True:
        for _, batch in enumerate(train_loader):
            batch_idx += 1
            start_time = time.time()
            batch['query'] = to_device(batch['query'], model.device)
            batch['pos'] = to_device(batch['pos'], model.device)

            if batch['neg'] is not None:
                batch['neg'] = to_device(batch['neg'], model.device)

            if args.accum_freq == 1: # no accum
                loss_dict, features_dict = model(batch['query'], batch['pos'], batch['neg'], batch['mode'])
                total_loss, cl_loss = loss_dict['total_loss'], loss_dict['cl_loss']
                accelerator.backward(total_loss)

            else: # gradient accumulation
                with torch.no_grad():
                    query_outputs = model(qry=batch['query'])
                    query_features = query_outputs['qry_reps']
                    query_features2 = query_outputs['qry_reps2']
                    pos_outputs = model(tgt=batch['pos'])
                    pos_features = pos_outputs['tgt_reps']
                    pos_features2 = pos_outputs['tgt_reps2']
                    neg_features = model(neg=batch['neg'])['neg_reps']
                    accum_features['query'].append(query_features)
                    accum_features['query2'].append(query_features2)
                    accum_features['pos'].append(pos_features)
                    accum_features['pos2'].append(pos_features2)
                    accum_features['neg'].append(neg_features)

                    accum_inputs.append(batch)

                if (batch_idx + 1) % args.accum_freq > 0:
                    continue

                for accum_idx in range(args.accum_freq):
                    input_batch = accum_inputs[accum_idx]
                    query_outputs = model(qry=input_batch['query'])
                    pos_outputs = model(tgt=input_batch['pos'])
                    neg_outputs = model(neg=input_batch['neg'])
                    
                    input_query_features = torch.cat(accum_features['query'][:accum_idx] + [query_outputs['qry_reps']] + accum_features['query'][accum_idx+1:])
                    input_query_features2 = torch.cat(accum_features['query2'][:accum_idx] + [query_outputs['qry_reps2']] + accum_features['query2'][accum_idx+1:])
                    input_pos_features = torch.cat(accum_features['pos'][:accum_idx] + [pos_outputs['tgt_reps']] + accum_features['pos'][accum_idx+1:])
                    input_pos_features2 = torch.cat(accum_features['pos2'][:accum_idx] + [pos_outputs['tgt_reps2']] + accum_features['pos2'][accum_idx+1:])
                    if neg_outputs['neg_reps'] is not None:
                        input_neg_features = torch.cat(accum_features['neg'][:accum_idx] + [neg_outputs['neg_reps']] + accum_features['neg'][accum_idx+1:])
                    else:
                        input_neg_features = None

                    #input_query_batch = input_batch['query']
                    #input_pos_batch = input_batch['pos']

                    loss_dict = model.calculate_loss(
                                                     input_query_features,
                                                     input_query_features2,
                                                     input_pos_features,
                                                     input_pos_features2,
                                                     input_neg_features,
                                                     None,
                                                     None,
                                                     )

                    total_loss, cl_loss = loss_dict['total_loss'], loss_dict['cl_loss']
                    total_loss /= args.accum_freq
                    accelerator.backward(total_loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.accum_freq > 1:
                accum_inputs, accum_features = [], {'query': [], 'query2': [], 'pos': [], 'pos2': [], 'neg': []}

            if accelerator.is_main_process:
                log_dict = {"losses/total_loss": loss_dict['total_loss'],
                            "losses/cl_loss": loss_dict['cl_loss'],
                            "losses/loss_L1": loss_dict['loss_L1'],
                            "losses/loss_L2": loss_dict['loss_L2'],
                            "losses/loss_L3": loss_dict['loss_L3'],
                            "losses/loss_L4": loss_dict['loss_L4'],
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            }
                accelerator.log(log_dict, step=global_step)
                print('logged')

            if accelerator.is_local_main_process:
                print(f"Step: {global_step}, Total_loss: {total_loss.detach().item() * args.accum_freq:.3f}, CL_loss: {cl_loss.detach().item() if isinstance(cl_loss, torch.Tensor) else 0.0:.3f}, Time: {time.time() - start_time:.2f}")

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
                if global_step != 0 and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, 'checkpoint-%07d' % global_step)
                    unwrapped_model = accelerator.unwrap_model(model)
                    if 'zero3' in args.deepspeed:
                        lora_params = [p for name, p in unwrapped_model.named_parameters()]
                        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
                            if accelerator.is_main_process:
                                unwrapped_model.save(save_path)
                                print(f"{os.path.basename(save_path)} saved!")
                    else:
                        if accelerator.is_main_process:
                            unwrapped_model.save(save_path)
                            print(f"{os.path.basename(save_path)} saved!")

        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, 'checkpoint-epoch-%03d' % n_epoch)
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped_model.save(save_path)
                print(f"{os.path.basename(save_path)} saved!")
        n_epoch += 1

if __name__ == "__main__":
    main()
