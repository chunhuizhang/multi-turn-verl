# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs


# add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):

        story = example.pop("story")
        question_raw = example.pop("question")
        data_source = example.pop("data_source")
        answer = example['answer']
        question = f'Story: {story}\nQuestion: {question_raw}'

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "tom",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/tom_sft")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    
    train_ds = load_dataset('parquet', data_files='./data/raw_tom/ToM_train_HiEx_3200.parquet')['train']
    test_ds = load_dataset('parquet', data_files='./data/raw_tom/ToM_test_HiExTi.parquet')['train']

    train_dataset = train_ds.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_ds.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
