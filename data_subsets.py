from datasets import load_dataset
import json
import numpy as np
import torch

def get_nectar_subset(subset_size=1000):
    ds = load_dataset('berkeley-nest/Nectar')
    ds = ds.shuffle(seed=17)
    ds['train'] = ds['train'].select(range(subset_size))
    return ds

def get_helpsteer2_subset(subset_size=1000):
    # https://huggingface.co/datasets/nvidia/HelpSteer2/blob/main/disagreements/disagreements.jsonl.gz
    with open('disagreements.jsonl') as f:
        ds = np.array([json.loads(line) for line in f])

    np.random.seed(17)
    subset_mask = np.random.permutation(torch.arange(len(ds)))[:subset_size].flatten()
    return ds[subset_mask]
