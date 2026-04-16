'''
MuCo
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0
'''
import os
from typing import Dict
import torch
import torch.distributed as dist
from torch.distributed.nn import all_gather
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.model.processor import QWEN2_VL, get_backbone_name, print_master, backbone2model, VLM_IMAGE_TOKENS

from src.model.modeling_custom import CustomQwen2VLForConditionalGeneration
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 margin: float = 0.0,
                 use_gen_loss: bool = False,
                 token_bot: str = None,
                 token_eos: str = None,
                 pretraining: bool = False,
                 is_zero3: bool = False,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.use_gen_loss = use_gen_loss
        self.token_bot = token_bot
        self.token_eos = token_eos
        self.pretraining = pretraining
        self.is_zero3 = is_zero3
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs = None):
        gradient_checkpointing_kwargs={'use_reentrant': self.is_zero3}
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def encode_input(self, input):
        outputs = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled_output, pooled_aux_output = self._pooling(hidden_states, input['input_ids'], input['attention_mask'])
        return pooled_output, pooled_aux_output

    def _pooling(self, last_hidden_state, input_ids, attention_mask):
        if self.pretraining:
            # specific indexing
            row_indices, col_indices = torch.nonzero(torch.eq(input_ids, self.pooling[0]), as_tuple=True)
            reps_all = last_hidden_state[row_indices, col_indices]

            if self.process_rank % 8 == 0:
                    print('pretraining pooling all!')
            reps = [reps_all]
            reps2 = None
        elif type(self.pooling) == int:
            # specific indexing
            row_indices, col_indices = torch.nonzero(torch.eq(input_ids, self.pooling), as_tuple=True)
            reps = last_hidden_state[row_indices, col_indices]
        elif isinstance(self.pooling, list):
            # specific indexing
            row_indices, col_indices = torch.nonzero(torch.eq(input_ids, self.pooling[0]), as_tuple=True)
            reps_all = last_hidden_state[row_indices, col_indices]
            if reps_all.shape[0] == input_ids.shape[0] * 2:
                if self.process_rank % 8 == 0:
                    print('aux pooling!')
                reps, reps2 = reps_all[0::2], reps_all[1::2]
            else:
                reps = reps_all
                reps2 = None
        else:
            raise NotImplementedError

        if self.normalize:
            if isinstance(reps, list):
                for idx in range(len(reps)):
                    reps[idx] = torch.nn.functional.normalize(reps[idx], p=2, dim=-1)
            else:
                reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
            if reps2 is not None:
                reps2 = torch.nn.functional.normalize(reps2, p=2, dim=-1)
        return reps, reps2

    def build_mlp(self, hidden_size, projector_dim, z_dim):
        return nn.Sequential(
                    nn.Linear(hidden_size, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, z_dim),
                )

    @classmethod
    def build(cls, args, tokenizer=None, special_tokens=None, **kwargs):
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=args.hf_cache_dir)
        # Loading the base model
        # Qwen2-VL
        config._attn_implementation = "flash_attention_2"
        config.use_cache = False
        base_model = CustomQwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=args.hf_cache_dir,
            use_bidirectional=args.use_bidirectional,
        )

        if special_tokens:
            # add special tokens
            base_model.resize_token_embeddings(max(len(tokenizer), base_model.model.language_model.embed_tokens.num_embeddings))
            trainable_token_indices = {'embed_tokens': tokenizer.convert_tokens_to_ids(special_tokens)}
            args.pooling = trainable_token_indices['embed_tokens']
        else:
            trainable_token_indices = None

        if args.mean_pooling:
            args.pooling = 'mean'

        if args.lora:
            target_modules = []
            for name, module in base_model.named_modules():
                if isinstance(module, torch.nn.Linear) and "lm_head" not in name and 'visual' not in name:
                    target_modules.append(name)
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                trainable_token_indices=trainable_token_indices,
                lora_dropout=args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)

            model = cls(
                encoder=lora_model,
                pooling=args.pooling,
                normalize=args.normalize,
                temperature=args.temperature,
                margin=args.margin,
                use_gen_loss=args.use_gen_loss,
                token_bot=tokenizer.encode('assistant')[0] if tokenizer is not None else None,
                token_eos=tokenizer.encode('<|im_end|>')[0] if tokenizer is not None else None,
                pretraining=args.pretraining,

                is_zero3='zero3' in args.deepspeed,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=args.pooling,
                normalize=args.normalize,
                temperature=args.temperature,
                margin=args.margin,
                use_gen_loss=args.use_gen_loss,
                token_bot=tokenizer.encode('assistant')[0] if tokenizer is not None else None,
                token_eos=tokenizer.encode('<|im_end|>')[0] if tokenizer is not None else None,
                pretraining=args.pretraining,

                is_zero3='zero3' in args.deepspeed,
            )

        if args.gradient_checkpointing:
            base_model.enable_input_require_grads()

        return model

    def load(self, checkpoint_path, args, is_trainable=True):
        # Loading the base model
        model_name_or_path = checkpoint_path

        # Building the model on top of the base
        if args.lora: # it should be always true because we currently do not care about full-training.
            print_master(f'Loading LoRA from {model_name_or_path}')

            self.encoder.load_adapter(model_name_or_path, self.encoder.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
               self.encoder = self.encoder.merge_and_unload()
        else:
            print_master(f'Loading the trained model from {model_name_or_path}')

            self.encoder = CustomQwen2VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
            )

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, neg: Dict[str, Tensor] = None, mode = None, *args, **kwargs):
        qry_reps, qry_aux = self.encode_input(qry) if qry else (None, None) # (bsz_per_device, dim)
        tgt_reps, tgt_aux = self.encode_input(tgt) if tgt else (None, None) # (bsz_per_device, dim)
        neg_reps, neg_aux = self.encode_input(neg) if neg else (None, None)

        features_dict = {"qry_reps": qry_reps, "qry_reps2": qry_aux, 
                         "tgt_reps": tgt_reps, "tgt_reps2": tgt_aux,
                         "neg_reps": neg_reps}
        if qry_reps is None or tgt_reps is None:
            return features_dict

        loss_dict = self.calculate_loss(qry_reps, qry_aux, tgt_reps, tgt_aux, neg_reps, neg_aux, mode)

        return loss_dict, features_dict

    def create_asymmetric_intra_class_mask(self,
                                           query_labels: torch.Tensor,
                                           key_labels: torch.Tensor,
                                           query_indices: torch.Tensor,
                                           key_indices: torch.Tensor,
                                           dtype: torch.dtype,
                                           ) -> torch.Tensor:
        same_class_mask = query_labels.unsqueeze(1) == key_labels.unsqueeze(0)

        positive_pair_mask = query_indices.unsqueeze(1) == key_indices.unsqueeze(0)

        mask_to_apply = same_class_mask & ~positive_pair_mask

        final_mask = torch.zeros_like(mask_to_apply, dtype=dtype)
        final_mask.masked_fill_(mask_to_apply, float('-inf'))
        
        return final_mask

    def calculate_loss(self, qry_reps, qry_aux, tgt_reps, tgt_aux, neg_reps, neg_aux, mode):
        if self.pretraining:
            cl_loss = 0
            for idx in range(len(qry_reps)):
                curr_qry_reps = qry_reps[idx]
                curr_tgt_reps = tgt_reps[idx]

                all_curr_qry_reps = self._dist_gather_tensor(curr_qry_reps)
                all_curr_tgt_reps = self._dist_gather_tensor(curr_tgt_reps)

                scores_per_query = torch.matmul(curr_qry_reps, all_curr_tgt_reps.transpose(0, 1))

                scores_per_pos = torch.matmul(curr_tgt_reps, all_curr_qry_reps.transpose(0, 1))

                target = torch.arange(scores_per_query.size(0), device=scores_per_query.device) + scores_per_query.size(0) * self.process_rank

                n = 7
                query_indices = target
                key_size = scores_per_query.shape[1]
                key_indices = torch.arange(key_size, device=target.device)
                query_labels = query_indices // n
                key_labels = key_indices // n
                intra_class_mask = self.create_asymmetric_intra_class_mask(
                    query_labels, 
                    key_labels, 
                    query_indices, 
                    key_indices,
                    scores_per_query.dtype
                )

                scores_per_query += intra_class_mask
                scores_per_pos += intra_class_mask

                cl_loss += self.cross_entropy(scores_per_query / self.temperature, target)

            total_loss = cl_loss
            loss_l1, loss_l2, loss_l3, loss_l4 = 0, 0, 0, 0

        else:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_qry_aux = self._dist_gather_tensor(qry_aux)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_tgt_aux = self._dist_gather_tensor(tgt_aux)

            all_neg_reps = self._dist_gather_tensor(neg_reps)


            scores_per_query, scores_per_pos, scores_per_query2, scores_per_pos2, scores_per_query3, scores_per_pos3, scores_per_query4, scores_per_pos4 \
                    = self.compute_similarity(qry_reps, all_qry_reps, qry_aux, all_qry_aux, tgt_reps, all_tgt_reps, tgt_aux, all_tgt_aux, all_neg_reps)

            target = torch.arange(scores_per_query.size(0), device=scores_per_query.device) + scores_per_query.size(0) * self.process_rank

            loss_l1 = self.cross_entropy(scores_per_query / self.temperature, target)
            loss_l2 = self.cross_entropy(scores_per_query2 / self.temperature, target)
            loss_l3 = self.cross_entropy(scores_per_query3 / self.temperature, target)
            loss_l4 = self.cross_entropy(scores_per_query4 / self.temperature, target)

            cl_loss = loss_l1 + loss_l2 + loss_l3 + loss_l4

            total_loss = cl_loss

        return {'total_loss': total_loss, 'cl_loss': cl_loss, 'loss_L1': loss_l1, 'loss_L2': loss_l2, 'loss_L3': loss_l3, 'loss_L4': loss_l4}

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = torch.cat(all_gather(t), dim=0) # gather with gradient
        return all_tensors

    def compute_similarity(self, local_query, global_query, local_query_aux, global_query_aux, local_pos, global_pos, local_pos_aux, global_pos_aux, global_neg):
        '''
        query           pos
        query_aux       pos_aux

        1. query vs pos
        2. query vs pos_aux
        3. query_aux vs pos
        4. query_aux vs pos_aux
        '''
        # 1
        logits_per_query = torch.matmul(local_query, global_pos.transpose(0, 1))
        logits_per_pos = torch.matmul(local_pos, global_query.transpose(0, 1))
        # 2
        logits_per_query2 = torch.matmul(local_query, global_pos_aux.transpose(0, 1))
        logits_per_pos2 = torch.matmul(local_pos_aux, global_query.transpose(0, 1))
        # 3
        logits_per_query3 = torch.matmul(local_query_aux, global_pos.transpose(0, 1))
        logits_per_pos3 = torch.matmul(local_pos, global_query_aux.transpose(0, 1))
        # 4
        logits_per_query4 = torch.matmul(local_query_aux, global_pos_aux.transpose(0, 1))
        logits_per_pos4 = torch.matmul(local_pos_aux, global_query_aux.transpose(0, 1))

        # handling negative pairs
        if global_neg is not None:
            print_master('hn')
            logits_query_neg = torch.matmul(local_query, global_neg.transpose(0, 1))
            logits_query_aux_neg = torch.matmul(local_query_aux, global_neg.transpose(0, 1))

            logits_per_query_output = torch.cat([logits_per_query, logits_query_neg], dim=-1)
            logits_per_query2_output = torch.cat([logits_per_query2, logits_query_neg], dim=-1)
            logits_per_query3_output = torch.cat([logits_per_query3, logits_query_aux_neg], dim=-1)
            logits_per_query4_output = torch.cat([logits_per_query4, logits_query_aux_neg], dim=-1)
        else:
            logits_per_query_output = logits_per_query
            logits_per_query2_output = logits_per_query2
            logits_per_query3_output = logits_per_query3
            logits_per_query4_output = logits_per_query4

        logits_per_pos_output = logits_per_pos
        logits_per_pos2_output = logits_per_pos2
        logits_per_pos3_output = logits_per_pos3
        logits_per_pos4_output = logits_per_pos4

        return logits_per_query_output, logits_per_pos_output, logits_per_query2_output, logits_per_pos2_output, logits_per_query3_output, logits_per_pos3_output, logits_per_query4_output, logits_per_pos4_output

    def get_labels(self, input_ids, token_bot_id, token_eos_id):
        batch_size, seq_len = input_ids.shape

        # 1. 모든 labels를 -100으로 초기화
        masked_labels = torch.full_like(input_ids, -100)

        for i in range(batch_size):
            current_input_ids = input_ids[i]

            bot_starts = (current_input_ids == token_bot_id).nonzero(as_tuple=True)[0]

            for idx, start_idx in enumerate(bot_starts):
                start_idx += 1
                end_idx_candidates = (current_input_ids[start_idx:] == token_eos_id).nonzero(as_tuple=True)[0]

                if end_idx_candidates.numel() > 0:
                    end_idx = start_idx + end_idx_candidates[0].item() + 1
                else:
                    end_idx = seq_len

                masked_labels[i, start_idx:end_idx] = current_input_ids[start_idx:end_idx]

        return masked_labels
