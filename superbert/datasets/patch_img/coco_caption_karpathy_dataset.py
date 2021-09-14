from superbert.datasets.ori_img.base_dataset import BaseDataset
from superbert.datasets.builder import DATASETS
import numpy as np
import torch


@DATASETS.register_module("patch_coco")
class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ["coco_caption_karpathy_test"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        # image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        # image_bytes.seek(0)
        path = self.table[image_key][index].as_py()
        image_tensor = torch.from_numpy(np.load(path.replace('.jpg', '.npy'))).squeeze(0)
        return image_tensor
    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split:
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite
