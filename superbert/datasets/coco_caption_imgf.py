from re import I
from .base_dataset import BaseDataset
from .builder import DATASETS
import torch
import pyarrow as pa
import os
import pdb
import json
import numpy as np
import cv2
import random
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

@DATASETS.register_module("coco_imf")
class CocoCaptionImfDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):

        self.text_column_name = kwargs['text_column_name']
        remove_duplicate = kwargs['remove_duplicate']
        self.draw_false_image = kwargs['draw_false_image']
        self.draw_false_text = kwargs['draw_false_text']
        split = kwargs['split']
        assert split in ["train", "val", "test"]

        self.tokenizer = kwargs['tokenizer']
        collator = (
            DataCollatorForWholeWordMask
            if kwargs["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=kwargs["mlm"], mlm_probability=kwargs["mlm_prob"]
        )
        # pdb.set_trace()
        self.max_text_len = kwargs['max_text_len']

        self.split = split
        if split == "train":
            names = ["coco_caption_imf_train"]
        elif split == "val":
            names = ["coco_caption_imf_val"]
        elif split == "test":
            names = ["coco_caption_imf_test"]

        self.data_dir = kwargs['data_dir']
        assert type(self.data_dir) is list
        self.all_data = []
        for data_cur in self.data_dir:
            with open(data_cur, 'r') as f:
                data_cur_info = json.load(f)
                for line in data_cur_info:
                    self.all_data.append(line)
        # pdb.set_trace()
        self.names = names

    def get_false_image(self, rep, single_id):
        random_index = random.randint(0, len(self.all_data) - 1)
        while random_index == single_id:
            random_index = random.randint(0, len(self.all_data) - 1)
        imf_f = self.all_data[random_index]
        res_dict = self.get_image_feature(imf_f[2])
        return {f"false_image_{rep}": res_dict}

    def get_image_feature(self, arrow_file = ''):
        res_dict = {}
        tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(arrow_file, "r")
                ).read_all()
            ]
        single_table = pa.concat_tables(tables, promote=True)
        # single_id = single_table['image_id'].to_pandas().tolist()[0]
        # dict_keys(['image_id', 'image_path', 'boxes', 'box_features', 'attr_feature', 'captions'])
        for text_column in self.text_column_name:
            tmp_cur = single_table[text_column].to_pandas().tolist()[0]
            if type(tmp_cur) is np.ndarray:
                res_dict[text_column] = np.array(tmp_cur.tolist())
            else:
                res_dict[text_column] = tmp_cur
        return res_dict

    def get_false_text(self, rep, single_id):
        random_index = random.randint(0, len(self.all_data) - 1)
        while random_index == single_id:
            random_index = random.randint(0, len(self.all_data) - 1)
        imf_f = self.all_data[random_index]
        txt = self.get_text(imf_f[0], imf_f[1], imf_f[3])
        return {f"false_text_{rep}": txt['text']}

    def get_text(self, key, path, text = ''):
        text = text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {'text': (key, path, text, encoding)}

    def __getitem__(self, index):
        imf_f = self.all_data[index]
        res_dict = {}
        res_dict.update({'image': self.get_image_feature(imf_f[2])})
        txt = self.get_text(imf_f[0], imf_f[1], imf_f[3])
        res_dict.update(txt)
        for i in range(self.draw_false_image):
            res_dict.update(self.get_false_image(i, res_dict['image']['image_id']))
        for i in range(self.draw_false_text):
            res_dict.update(self.get_false_text(i, res_dict['image']['image_id']))
        # pdb.set_trace()
        return res_dict

    def __len__(self):
        return len(self.all_data)

    def collate(self, batch):
        result_dict_batch = {}
        batch_size = len(batch)     
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # dict_batch['text'][1]

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

        img_sizes = list()
        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [i['box_features'].shape for i in img]

        for size in img_sizes:
            assert (
                len(size) == 2
            ), f"Collate error, an image should be in shape of (H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[0] for i in img_sizes]) # box num
            max_width = max([i[1] for i in img_sizes])  # 2048 512
        for img_key in img_keys:
            img = dict_batch[img_key]
            new_images = torch.zeros(batch_size, max_height, max_width) # max_width: 2048
            new_images_attr = torch.zeros(batch_size, max_height, 512)  # 512
            for bi in range(batch_size):
                orig = img[bi]['box_features']
                orig_attr = img[bi]['attr_feature']
                new_images[bi, : orig.shape[0], : orig.shape[1]] = torch.Tensor(orig)
                new_images_attr[bi, : orig_attr.shape[0], : orig_attr.shape[1]] = torch.Tensor(orig_attr)
            # dict_batch[img_key + '_batch'] = new_images
            result_dict_batch[img_key + '_boxf_batch'] = new_images
            result_dict_batch[img_key + '_attr_batch'] = new_images_attr

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        texts       = [[caption[2] for caption in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings   = [[caption[3] for caption in dict_batch[txt_key]] for txt_key in txt_keys]
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = self.mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):  # 'false_text_0', 'text'
            texts, encodings = (
                [d[2] for d in dict_batch[txt_key]],
                [d[3] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            token_type_ids = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask ,_token_type_ids= (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                        torch.tensor(encoding["token_type_ids"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask
                    token_type_ids[_i, : len(_token_type_ids)] = _token_type_ids
            result_dict_batch[txt_key] = texts
            result_dict_batch[f"{txt_key}_ids"] = input_ids
            result_dict_batch[f"{txt_key}_segment_ids"] = token_type_ids
            result_dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            result_dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            result_dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            result_dict_batch[f"{txt_key}_masks"] = attention_mask            
            # pdb.set_trace()
            
        # pdb.set_trace()
        return result_dict_batch

    def get_pretrained_tokenizer(self, from_pretrained, cache_dir):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, cache_dir=cache_dir, do_lower_case="uncased" in from_pretrained
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, do_lower_case="uncased" in from_pretrained
        )