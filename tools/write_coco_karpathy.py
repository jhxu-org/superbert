import json
import os
from PIL.Image import NONE
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
import gc


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    # with open(path, "rb") as fp:
    #     binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [path, captions, name, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)

    with open(f"{root}/v2_mscoco_train2014_annotations.json", "r") as fp:
        tran_anno = json.load(fp)
    with open(f"{root}/v2_mscoco_val2014_annotations.json", "r") as fp:
        val_anno = json.load(fp)
    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/train2014/*.jpg")) + list(glob(f"{root}/val2014/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    # bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]
    os.makedirs(dataset_root, exist_ok=True)

    for split in ["train", "val", "restval", "test"]:
        with pa.OSFile(f"{dataset_root}/coco_caption_karpathy_{split}.arrow", "wb"
                        ) as sink:
            writer = None
            for path in tqdm(caption_paths):
                name = path.split("/")[-1]
                if iid2split[name] == split:
                    bs = path2rest(path, iid2captions, iid2split)

                    dataframe = pd.DataFrame(
                        [bs], columns=["image", "caption", "image_id", "split"],
                    )

                    table = pa.Table.from_pandas(dataframe)
                    if writer == None:
                        writer = pa.RecordBatchFileWriter(sink, table.schema)
                    # with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            writer.close()



if __name__ == "__main__":
    make_arrow('/search/gpu3/xujianhua/data/coco', '/search/gpu1/superbert_roi')