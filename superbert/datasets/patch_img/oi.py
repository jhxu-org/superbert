from posixpath import join
import random
import torch
import io
import pyarrow as pa
import os
from PIL import Image
import numpy as np
import re
import csv
from shutil import copyfile
import cv2
from tqdm import tqdm
import numpy as np
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
from superbert.datasets.builder import DATASETS

@DATASETS.register_module("patch_oi")
class OIv5Dataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        
        split = kwargs['split']
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["train"]
        elif split == "val":
            names = ["validation"]
        elif split == "test":
            names = ["test"]
        self.data_dir = kwargs['data_dir']
        self.transforms = kwargs['transforms']
        self.names = names
        self.max_text_len = kwargs['max_text_len']
        self.image_only = kwargs['image_only']
        self.cache_dir = os.path.join(kwargs['cache_dir'], 'oi', split)
        self.annos = {}
        self.cls_label = {}
        self.tokenizer = kwargs['tokenizer']#self.get_pretrained_tokenizer(kwargs['tokenizer'],kwargs['cache_dir'])
        for name in names:
            with open(os.path.join(self.data_dir, "oidv6-class-descriptions.csv"), encoding='utf-8') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', )
                for idx, row in enumerate(tqdm(spamreader)):
                    if idx == 0:
                        continue
                    self.cls_label[row[0]] = row[1]

        if not os.path.exists(self.cache_dir):
            if len(names) != 0:
                for name in names:
                    # with open(os.path.join(self.data_dir, "oidv6-class-descriptions.csv"), encoding='utf-8') as csvfile:
                    #     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', )
                    #     for idx, row in enumerate(tqdm(spamreader)):
                    #         if idx == 0:
                    #             continue
                    #         self.cls_label[row[0]] = row[1]
                    with open(os.path.join(self.data_dir, "anno/box", f"{name}-annotations-bbox.csv"), newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for idx, row in enumerate(tqdm(spamreader)):
                            if idx == 0:
                                continue
                            # if idx > 1000:
                            #     break
                            key = row[0]
                            if not key in self.annos:
                                self.annos[key] = {}
                                self.annos[key]['boxes'] = []
                                self.annos[key]['cls_key'] = set()
                            self.annos[key]['path'] = os.path.join(self.data_dir,name, f"{key}.jpg")
                            self.annos[key]['boxes'].append([self.cls_label[row[2]], int(row[3]), float(row[4]),float(row[5]),float(row[6]),float(row[7])])
                            self.annos[key][self.cls_label[row[2]]] =  self.annos[key].get(self.cls_label[row[2]], 0) + 1
                            self.annos[key]['cls_key'].add(self.cls_label[row[2]]) 
            self.image_ids = list(self.annos.keys())
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(os.path.join(self.cache_dir, "annos.npy"),self.annos)
            np.save(os.path.join(self.cache_dir, "image_ids.npy"),self.image_ids)
        else:
            print(f"********** loading {split} npy **********")
            self.annos = np.load(os.path.join(self.cache_dir, "annos.npy"),allow_pickle=True).item()
            self.image_ids = np.load(os.path.join(self.cache_dir, "image_ids.npy"),allow_pickle=True).tolist()
            print(f"********** loading {split} done **********")

        self.mlm_collator = DataCollatorForLanguageModeling(self.tokenizer, False, 0)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

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
    # def _get_dets_to_use(self,sentence, objects):
    #     """
    #     We might want to use fewer detectiosn so lets do so.
    #     :param item:
    #     :param question:
    #     :param answer_choices:
    #     :return:
    #     """
    #     # Load questions and answers
    #     dets = re.findall(r"##\d+",sentence)
    #     dets
        
    #     return dets2use, old_det_to_new_ind


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        anno = self.annos[img_id]
        image_tensor = np.load(anno['path'].replace('/search/gpu4/renyili/dataset', 
                    '/search/gpu3/xujianhua/data/dino_patch').replace('.jpg', '.npy'))
        # image = Image.open(anno['path']).convert("RGB")
        # image_tensor = self.transforms(image)#[tr(image) for tr in self.transforms]

        test_class_size = random.randint(1,20)
        if test_class_size < len(anno['cls_key']):
            sample_clskey = random.choices(list(anno['cls_key']), k=test_class_size)
        else:
            sample_clskey = random.choices([_cls for _, _cls in self.cls_label.items() if _cls not in anno['cls_key']],
                k=test_class_size-len(anno['cls_key']))
            sample_clskey += anno['cls_key']
        sample_clslabel = [anno.get(_cls, 0) for _cls in sample_clskey]
        assert len(sample_clslabel) == len(sample_clskey)
        text = "[SEP][CLS]".join(sample_clskey)
        # labels = self.table["labels"][index].as_py()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        return {
            "image": [image_tensor],
            "text": (text, encoding),
            "cls_labels": sample_clslabel,
        }

    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
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
                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_segment_ids"] = token_type_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask


                

        return dict_batch