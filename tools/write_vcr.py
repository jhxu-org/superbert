import sys
import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas] if "test" not in split else list(list())
    answer_labels = (
        [a["labels"] for a in answers] if "test" not in split else list(list())
    )
    answer_scores = (
        [a["scores"] for a in answers] if "test" not in split else list(list())
    )
    answers = (
        [[label2ans[l] for l in al] for al in answer_labels]
        if "test" not in split
        else list(list())
    )

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]

def replace_tokens(tokens):
    new_tokens = []
    for tok in tokens:
        if isinstance(tok, list):
            tok = '##'+str(tok[0])
        new_tokens.append(tok)
    return ' '.join(new_tokens)
def make_arrow(root, dataset_root):
    with open(f"{root}/train.jsonl", "r") as fp:
        train_items = [json.loads(s) for s in fp]
    with open(f"{root}/test.jsonl", "r") as fp:
        test_items = [json.loads(s) for s in fp]
    with open(f"{root}/val.jsonl", "r") as fp:
        val_items = [json.loads(s) for s in fp]

    annotations = dict()

    for split, datainfo in zip(
        ["train", "val", "test"],
        [
            train_items,
            val_items,
            test_items,
        ],
    ):
        annotations[split] = datainfo


    for split in [
        "train",
        "val",
        "test",
    ]:
        annots = annotations[split]
        random.shuffle(annots)
        
        print(
            "data size:",
            
            len(annots)
        )
        
        os.makedirs(dataset_root, exist_ok=True)

        with pa.OSFile(f"{dataset_root}/vcr_{split}.arrow", "wb") as sink:
            writer = None
            for idx, annot in enumerate(tqdm(annots)):
                # bs = [
                #     path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
                # ]
                iid = idx

                with open(f"{root}/vcr1images/{annot['img_fn']}", "rb") as fp:
                    binary = fp.read()
                with open(f"{root}/vcr1images/{annot['metadata_fn']}", 'r') as f:
                    metadata = json.load(f)
                qids = annot['annot_id']
                questions = replace_tokens(annot['question'])
                # questions_orig = replace_tokens(annot['question_orig'])

                answer_choices = [replace_tokens(an) for an in annot['answer_choices']]
                answer_label = annot['answer_label'] if "test" not in split else -1

                rationale_choices = [replace_tokens(an) for an in annot['rationale_choices']]
                rationale_label = annot['rationale_label'] if "test" not in split else -1
                boxes = metadata['boxes']
                objects = annot['objects']
                # annot_id = annot['annot_id']
                # answers = answer_choices[answer_label] if "test" not in split else ""
                bs = [binary, questions,answer_choices, answer_label, rationale_choices, rationale_label,boxes, objects, iid, qids, split]
                dataframe = pd.DataFrame(
                    [bs],
                    columns=[
                        "image",
                        "questions",
                        "answer_choices",
                        "answer_label",
                        "rationale_choices",
                        "rationale_label",
                        "boxes",
                        "objects",
                        "iid",
                        "qids",
                        "split"
                    ],
                )

                table = pa.Table.from_pandas(dataframe)
                if writer == None:
                    writer = pa.RecordBatchFileWriter(sink, table.schema)
                
                writer.write_table(table)

            writer.close()



if __name__ == "__main__":
    make_arrow('/search/gpu4/vcr', '/search/gpu1/superbert_roi/')

