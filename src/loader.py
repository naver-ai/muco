'''
MuCo
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0
'''
import os
import glob
import random
import webdataset as wds
import torch
from torchvision import transforms
import transformers
from transformers import Qwen2VLProcessor
from transformers import Qwen2VLImageProcessor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from PIL import Image
import pandas as pd
import numpy as np

PHI3V = 'phi3_v'
QWEN2_VL = 'qwen2_vl'
VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    QWEN2_VL: "<|vision_start|><|image_pad|><|vision_end|>",
}
EMB_TOKEN = "<|emb|>"
MASK_TOKEN = "<|mask|>"

Backbone_names = {'Qwen/Qwen2-VL-2B-Instruct': 'qwen2_vl',
                  'Qwen/Qwen2-VL-7B-Instruct': 'qwen2_vl',
                  }

TEMPLATE_BASE = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_TRAIN = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
Please rewrite your last response in human-readable language.
<|im_start|>assistant
{gen_sentence}<|im_end|>
<|im_start|>user
Generate a response that first reconstructs your very last response by inferring any masked information, then acknowledges or rephrases the complete what I asked you, and finally integrates the Answer information seamlessly.<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_TRAIN_REPHRASE = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
Radically rephrase your last response to be fundamentally different in structure and vocabulary while preserving the core meaning.
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_PRETRAIN_SIGNLETURN = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence1}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_PRETRAIN_ = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence1}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence2}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence3}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence4}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence5}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence6}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_PRETRAIN = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence1}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence2}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence3}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence4}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence5}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence6}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence7}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_PRETRAIN_HN = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_sentence1}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence2}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence3}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence4}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence5}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>
<|im_start|>user
{user_sentence6}<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_ORACLE_QWEN = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Synthesize the following query and answer into a single, coherent response.

USER\'s Query: {user_sentence}
Answer: {gen_sentence}

Generate a response that first reconstructs the USER's Query by inferring any masked information from the Answer, then acknowledges or rephrases the complete query, and finally integrates the Answer information seamlessly.<|im_end|>
<|im_start|>assistant
<|emb|><|im_end|>'''

TEMPLATE_BASE_SIMPLE = '{user_sentence}'
TEMPLATE_ORACLE_SIMPLE = '''USER\'s Query: {user_sentence}
Answer: {gen_sentence}'''

TEMPLATE_BASE_SMOL = '<|im_start|>System: You are a helpful assistant.<end_of_utterance>\nUser: {user_sentence}<end_of_utterance>\nAssistant: <|emb|><end_of_utterance>'
TEMPLATE_ORACLE_SMOL = '''<|im_start|>System: You are a helpful assistant.<end_of_utterance>
User: Synthesize the following query and answer into a single, coherent response.

USER\'s Query: {user_sentence}
Answer: {gen_sentence}

Generate a response that first reconstructs the USER's Query by inferring any masked information from the Answer, then acknowledges or rephrases the complete query, and finally integrates the Answer information seamlessly.<end_of_utterance>
Assistant: <|emb|><end_of_utterance>'''


class CustomDataset(Dataset):
    def __init__(self, meta_list, image_folder, processor, special_tokens, vlm_image_token, template_train, template_pretrain, pretraining):
        self.meta_list = meta_list
        self.image_folder = image_folder
        self.processor = processor
        self.special_tokens = special_tokens
        self.vlm_image_token = vlm_image_token
        self.template_train = template_train
        self.template_pretrain = template_pretrain
        self.pretraining = pretraining

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]

        vlm_image_token = self.vlm_image_token #VLM_IMAGE_TOKENS[QWEN2_VL]

        if not isinstance(meta['qry'], str):
            if self.pretraining:
                # shuffle first!
                '''
                query idx:
                0: classification
                1, 2: Global VQA
                3, 4: Local VQA
                5: Creative VQA
                6: Retreival
                '''
                shuffled_idx = list(range(len(meta['qry'])))
                random.shuffle(shuffled_idx)

                query_text, pos_text, pos_sentence, neg_text = [], [], [], []
                for sample_idx in shuffled_idx:
                    sampled_query_text = meta['qry'][sample_idx]
                    if not isinstance(query_text, str):
                        sampled_query_text = random.sample(sampled_query_text.tolist(), k=1)[0]
                    query_text.append(sampled_query_text)
                    pos_text.append(meta['pos_text'][sample_idx])
                    pos_sentence.append(meta['pos_sentence'][sample_idx])
                    neg_text.append(meta['neg_text'][sample_idx])
                query_sentence = meta['qry_sentence']
            else:
                # select the only one
                sample_idx = random.sample(range(len(meta['qry'])), k=1)[0]
                query_text = meta['qry'][sample_idx]
                if not isinstance(query_text, str):
                    query_text = random.sample(query_text.tolist(), k=1)[0]
                pos_text = meta['pos_text'][sample_idx]
                neg_text = meta['pos_text'][sample_idx]
                query_sentence = meta['qry_sentence']
                pos_sentence = meta['pos_sentence'][sample_idx]
            neg_image_path = ''
        else:
            query_text = meta['qry']
            pos_text = meta['pos_text']
            query_sentence = meta['qry_sentence']
            pos_sentence = meta['pos_sentence']

            if 'neg_text' in meta:
                if type(meta['neg_text']) == np.ndarray:
                    neg_text_list = meta['neg_text']
                else:
                    neg_text_list = eval(meta['neg_text'])
                if type(meta['neg_image_path']) == np.ndarray:
                    neg_image_list = meta['neg_image_path']
                else:
                    neg_image_list = eval(meta['neg_image_path'])
                neg_sample_idx = random.sample(range(len(neg_text_list)), k=1)[0]
                neg_text = neg_text_list[neg_sample_idx]
                neg_image_path = neg_image_list[min(neg_sample_idx, len(neg_image_list)-1)]
            else:
                print('running without HN')
                neg_text = ''
                neg_image_path = ''

        if self.pretraining:
            for sample_idx in range(len(query_text)):
                if sample_idx == 0:
                    query_text[sample_idx] = query_text[sample_idx].replace(VLM_IMAGE_TOKENS[PHI3V], vlm_image_token)
                else:
                    query_text[sample_idx] = query_text[sample_idx].replace(f"{VLM_IMAGE_TOKENS[PHI3V]}\n", '')
            for sample_idx in range(len(pos_text)):
                if sample_idx == 0:
                    pos_text[sample_idx] = pos_text[sample_idx].replace(VLM_IMAGE_TOKENS[PHI3V], vlm_image_token)
                else:
                    pos_text[sample_idx] = pos_text[sample_idx].replace(f"{VLM_IMAGE_TOKENS[PHI3V]}\n", '')
        else:
            query_text = query_text.replace(VLM_IMAGE_TOKENS[PHI3V], vlm_image_token)
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], vlm_image_token)
            neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], vlm_image_token)

        query_image_path = meta['qry_image_path']
        pos_image_path = meta['pos_image_path']
        # neg_image_path is already defined above.

        if len(query_image_path) > 0:
            query_image_path = os.path.join(self.image_folder, query_image_path)
            query_image = Image.open(query_image_path).convert('RGB')
            if query_image.size[0] < 28 or query_image.size[1] < 28:
                query_image = query_image.resize((56, 56))
        else:
            if self.pretraining:
                for sample_idx in range(len(query_text)):
                    query_text[sample_idx] = query_text[sample_idx].replace(vlm_image_token, '')
            else:
                query_text = query_text.replace(vlm_image_token, '') # sanitizing
            query_image = None

        if len(pos_image_path) > 0:
            pos_image_path = os.path.join(self.image_folder, pos_image_path)
            pos_image = Image.open(pos_image_path).convert('RGB')
            if pos_image.size[0] < 28 or pos_image.size[1] < 28:
                pos_image = pos_image.resize((56, 56))
        else:
            if self.pretraining:
                for sample_idx in range(len(pos_text)):
                    pos_text[sample_idx] = pos_text[sample_idx].replace(vlm_image_token, '')
            else:
                pos_text = pos_text.replace(vlm_image_token, '')
            pos_image = None

        if len(neg_image_path) > 0:
            neg_image_path = os.path.join(self.image_folder, neg_image_path)
            neg_image = Image.open(neg_image_path).convert('RGB')
            if neg_image.size[0] < 28 or neg_image.size[1] < 28:
                neg_image = neg_image.resize((56, 56))
            if vlm_image_token not in neg_text:
                neg_text = f"{vlm_image_token}\n" + neg_text
        else:
            if self.pretraining:
                for sample_idx in range(len(neg_text)):
                    neg_text[sample_idx] = neg_text[sample_idx].replace(vlm_image_token, '')
            else:
                neg_text = neg_text.replace(vlm_image_token, '')
            neg_image = None

        if self.pretraining:
            query_text = self.template_pretrain.format(user_sentence1=query_text[0],
                                                       user_sentence2=query_text[1],
                                                       user_sentence3=query_text[2],
                                                       user_sentence4=query_text[3],
                                                       user_sentence5=query_text[4],
                                                       user_sentence6=query_text[5],
                                                       user_sentence7=query_text[6],
                                                       )
            pos_text = self.template_pretrain.format(user_sentence1=pos_text[0],
                                                     user_sentence2=pos_text[1],
                                                     user_sentence3=pos_text[2],
                                                     user_sentence4=pos_text[3],
                                                     user_sentence5=pos_text[4],
                                                     user_sentence6=pos_text[5],
                                                     user_sentence7=pos_text[6],
                                                     )
            neg_text = ''

            if query_image:
                query_image = self.resize_image(query_image, 640)
            if pos_image:
                pos_image = self.resize_image(pos_image, 640)
            if neg_image:
                neg_image = self.resize_image(neg_image, 640)
        else:
            if len(query_text) == 0:
                query_text = '\n'
            if len(pos_text) == 0:
                pos_text = '\n'
            if len(neg_text) == 0:
                neg_text = '\n'

            if query_text[-1] != '\n':
                query_text += '\n'
            if pos_text[-1] != '\n':
                pos_text += '\n'
            if neg_text[-1] != '\n':
                neg_text += '\n'

            gen_pos = pos_text #pos_sentence
            if pos_sentence != '':
                gen_pos = gen_pos.replace(vlm_image_token, f"Image description: {pos_sentence}")

            gen_query = query_text
            if query_sentence != '':
                gen_query = gen_query.replace(vlm_image_token, f"Image description: {query_sentence}")

            if random.random() < 0.5:
                gen_pos = MASK_TOKEN
            else:
                gen_pos = self.masking(gen_pos, 0.5)

            if random.random() < 0.5:
                gen_query = MASK_TOKEN
            else:
                gen_query = self.masking(gen_query, 0.5)

            query_text = self.template_train.format(user_sentence=query_text, gen_sentence=gen_pos)
            pos_text = self.template_train.format(user_sentence=pos_text, gen_sentence=gen_query)

            neg_text = TEMPLATE_BASE.format(user_sentence=neg_text)

        if query_image is None:
            query_mode = 0
        else:
            query_mode = 1
        if pos_image is None:
            pos_mode = 0
        else:
            pos_mode = 1

        return {'query_text': query_text,
                'query_image': query_image,
                'pos_text': pos_text,
                'pos_image': pos_image,
                'neg_text': neg_text,
                'neg_image': neg_image,
                'query_mode': query_mode,
                'pos_mode': pos_mode,
                }

    def masking(self, sentence, mask_prob):
        new_sentence = []
        for term in sentence.split(' '):
            if random.random() < mask_prob:
                if len(new_sentence) > 0 and new_sentence[-1] == MASK_TOKEN:
                    continue
                new_sentence.append(MASK_TOKEN)
            else:
                new_sentence.append(term)
        return ' '.join(new_sentence)

    def check_token(self, sentence, length):
        token = self.processor.tokenizer.encode(sentence)
        if len(token) > length:
            processed_sentence = self.processor.tokenizer.decode(token[:length])
            return processed_sentence
        return sentence

    def resize_image(self, pil_image, min_side_target):
        width, height = pil_image.size
        if width < height:
            # Width가 더 작은 경우 (세로형 이미지)
            scale_factor = min_side_target / width
            new_width = min_side_target
            new_height = int(height * scale_factor)
        else:
            # Height가 더 작거나 같은 경우 (가로형/정사각형 이미지)
            scale_factor = min_side_target / height
            new_height = min_side_target
            new_width = int(width * scale_factor)

        # 4. 이미지 리사이즈
        resized_img = pil_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        return resized_img

def base_collate_fn(batch, processor, make_image_first, model_name_or_path, pretraining):
    query_batch = {'text': [], 'images': []}
    pos_batch = {'text': [], 'images': []}
    neg_batch = {'text': [], 'images': []}
    mode_batch = {'query': [], 'pos': []}
    for sample in batch:
        for k, v in sample.items():
            if k == 'query_text':
                query_batch['text'].append(v)
            elif k == 'query_image':
                query_batch['images'].append(v)
            elif k == 'pos_text':
                if isinstance(v, list):
                    # handling the pretraining phase
                    pos_batch['text'] += v
                else:
                    pos_batch['text'].append(v)
            elif k == 'pos_image':
                pos_batch['images'].append(v)
            elif k == 'neg_text':
                neg_batch['text'].append(v)
            elif k == 'neg_image':
                neg_batch['images'].append(v)
            elif k == 'query_mode':
                mode_batch['query'].append(v)
            elif k == 'pos_mode':
                mode_batch['pos'].append(v)
            else:
                pass

    # do double-check the "image" of each batch.
    query_batch['images'] = [image for image in query_batch['images'] if image is not None]
    pos_batch['images'] = [image for image in pos_batch['images'] if image is not None]
    neg_batch['images'] = [image for image in neg_batch['images'] if image is not None]

    if len(query_batch['images']) == 0:
        query_batch['images'] = None
    if len(pos_batch['images']) == 0:
        pos_batch['images'] = None
    if len(neg_batch['images']) == 0:
        neg_batch['images'] = None

    max_length = None

    query_batch = processor(
        **query_batch,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    if make_image_first:
        rollback_processed_batch(query_batch, query_rollback_idx)

    pos_batch = processor(
        **pos_batch,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    neg_batch = processor(
        **neg_batch,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    if pretraining:
        neg_batch = None

    output_batch = {'query': query_batch, 'pos': pos_batch, 'neg': neg_batch, 'mode': mode_batch}

    return output_batch

def build_torch_dataloader(meta_folder, image_folder, n_samples, batch_size=2, add_emb_token=False, model_name='Qwen/Qwen2-VL-2B-Instruct', hf_cache_dir='./', pretraining=False, num_workers=8, is_train=True):
    # read meta files first
    meta_list = []
    for meta_path in sorted(glob.glob(os.path.join(meta_folder, '*/*_generated_sentence.parquet'))):
        current_meta_list = pd.read_parquet(meta_path).to_dict('records')
        if n_samples is not None:
            meta_list += current_meta_list[:min(n_samples, len(current_meta_list))]
        else:
            meta_list += current_meta_list

    # build a processor
    model_name_or_path = model_name

    min_pixels, max_pixels = 28*28*256, 28*28*1280
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, cache_dir=hf_cache_dir)
    processor = Qwen2VLProcessor.from_pretrained(model_name_or_path,
                                                 image_processor=image_processor,
                                                 min_pixels=min_pixels,
                                                 max_pixels=max_pixels,
                                                 cache_dir=hf_cache_dir,
                                                 )

    if add_emb_token:
        special_tokens = [EMB_TOKEN, MASK_TOKEN]
        processor.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    else:
        special_tokens = None

    # build a data loader
    dataset = CustomDataset(meta_list, image_folder, processor, special_tokens,
                            vlm_image_token=VLM_IMAGE_TOKENS[QWEN2_VL],
                            template_train=TEMPLATE_TRAIN if not pretraining else TEMPLATE_BASE,
                            template_pretrain=TEMPLATE_PRETRAIN,
                            pretraining=pretraining,
                            )

    collate_fn = partial(base_collate_fn, processor=processor, make_image_first=False, model_name_or_path=model_name_or_path, pretraining=pretraining)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return dataloader, processor.tokenizer, special_tokens

