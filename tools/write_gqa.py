import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2objects):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    # with open(path, "rb") as fp:
    #     binary = fp.read()

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]
    widths = [c["width"] for c in cdicts]
    heights = [c["height"] for c in cdicts]
    xs = [c["x"] for c in cdicts]
    ys = [c["y"] for c in cdicts]

    return [
        path.replace('/search/gpu3/xujianhua/data','/search/gpu3/xujianhua/data/dino_patch'),
        captions,
        widths,
        heights,
        xs,
        ys,
        str(iid),
        iid2objects[iid]
    ]


def make_arrow(root, dataset_root):
    with open(f"{root}/region_descriptions.json", "r") as fp:
        captions = json.load(fp)
    with open(f"{root}/objects.json", "r") as fp:
        objects = json.load(fp)

    iid2captions = defaultdict(list)
    for cap in tqdm(captions):
        cap = cap["regions"]
        for c in cap:
            iid2captions[c["image_id"]].append(c)

    iid2objects = defaultdict(list)
    for objs in tqdm(objects):
        iid2objects[objs["image_id"]] = objs["objects"]
        

    paths = list(glob(f"{root}/images/VG_100K/*.jpg")) + list(
        glob(f"{root}/images/VG_100K_2/*.jpg")
    )
    random.shuffle(paths)
    caption_paths = [
        path for path in paths if int(path.split("/")[-1][:-4]) in iid2captions
    ]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2objects) for path in tqdm(caption_paths)]
    dataframe = pd.DataFrame(
        bs, columns=["image", "caption", "width", "height", "x", "y", "image_id", "objects"],
    )
    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/vg.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
if __name__ == "__main__":
    make_arrow('/search/gpu4/renyili/dataset/VG', '/search/gpu3/superbert')
