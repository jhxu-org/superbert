import torch
import io
import pyarrow as pa
from PIL import Image
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import re
from superbert.datasets.builder import DATASETS
import random

@DATASETS.register_module("ori_vcr")
class VcrDataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dir, max_text_len, transforms,  **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_text_len = max_text_len
        self.transforms  = transforms
        self.tokenizer = kwargs['tokenizer']
        self.table = pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{self.data_dir}/vcr_{split}.arrow", "r")
                ).read_all()

        question = self.table['questions'].to_pandas().tolist()
        answer_choices = self.table['answer_choices'].to_pandas().tolist()
        qids = self.table['qids'].to_pandas().tolist()
        rationale_choices = self.table['rationale_choices'].to_pandas().tolist()
        objects = self.table['objects'].to_pandas().tolist()
        boxes = self.table['boxes'].to_pandas().tolist()
        if self.split != 'test':
            answer_label = self.table['answer_label'].to_pandas().tolist()
            rationale_label = self.table['rationale_label'].to_pandas().tolist()
        else:
            answer_label = [-1]*len(question)
            rationale_label = [-1]*len(question)
        self.all_texts = []
        for data in zip(qids, question, answer_choices,rationale_choices, answer_label, rationale_label, boxes, objects):
            self.all_texts.append(data)

    def __len__(self):
        return len(self.all_texts)

    def __getitem__(self, index):
        image_bytes = io.BytesIO(self.table['image'][index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        image_tensor = self.transforms(image)
        data = self.all_texts[index]
        qid = data[0]
        question = data[1]
        answer_choinces = data[2]
        rationale_choinces = data[3]
        texts = []
        if self.split != 'test':
            answer_label = data[4]
            rationale_label = data[5]
        else:
            answer_label = -1
            rationale_label = -1

        itm_labels = []
        for idx, answer in enumerate(answer_choinces):
            texts.append(question + answer)
            itm_labels.append(idx==answer_label)
        for idx in range(4):
            for idy, rationale in enumerate(rationale_choinces):
                texts.append(texts[idx] + rationale)
                itm_labels.append(idx==answer_label and idy==rationale_label)
        
        for idx in range(len(texts)):
            dets = [id.split("##")[1] for id in re.findall(r"(?:##)\d+", texts[idx])]
            if len(dets) != 0:
                objects = self.all_texts[index][7] 
                for det in dets:
                    texts[idx] = texts[idx].replace(r"##"+str(det), objects[int(det)])
        image_tensor = image_tensor
        if self.split != 'test':
            sels = random.choice([(txt, label) for txt, label in zip(texts, itm_labels)])
            texts = [sels[0]]
            itm_labels = sels[1]
            answer_label = data[4]
            rationale_label = data[5]
            image_tensor = [image_tensor]
        else:
            image_tensor = [image_tensor for i in range(20)]
            answer_label = -1
            rationale_label = -1

        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "image": image_tensor,
            "text": encoding,
            "ori":texts,
            "itm_labels":itm_labels,
            "answer_label":answer_label,
            "rationale_label":rationale_label,
            "qid": qid,
        }

    
    def collate(self, batch):
        batch_size = len(batch) * (len(batch[0]['image']))
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
            imgs = []
            for img in dict_batch[img_key]:
                imgs += img


            new_images = torch.zeros(batch_size, 3, max_height, max_width)

            for bi in range(batch_size):
                orig = imgs[bi]
                new_images[bi, :, : orig.shape[1], : orig.shape[2]] = orig
            # patch_imgs = new_images[0].unsqueeze(1).repeat(1,20,1,1,1)
            # patch_imgs = patch_imgs.reshape(-1, 3,patch_imgs.shape[-2],patch_imgs.shape[-1])
            dict_batch[img_key] = [new_images]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            encodings = [[d for d in dict_batch[txt_key]] for txt_key in txt_keys][0]
            dict_batch["text_ids"] = torch.cat([torch.tensor(enc["input_ids"]) for enc in encodings], dim=0)
            dict_batch["text_labels"] = torch.full_like(dict_batch["text_ids"], -100)#torch.cat([torch.tensor(enc["input_ids"]) for enc in encodings], dim=0)
            dict_batch["text_masks"] = torch.cat([torch.tensor(enc["attention_mask"]) for enc in encodings], dim=0)
            dict_batch[f"text_segment_ids"] = torch.cat([torch.tensor(enc["token_type_ids"]) for enc in encodings], dim=0)
            dict_batch["answer_label"] = torch.tensor(dict_batch["answer_label"])
            dict_batch["rationale_label"] = torch.tensor(dict_batch["rationale_label"])

        return dict_batch