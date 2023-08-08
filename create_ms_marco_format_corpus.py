from typing import List, Dict

import numpy as np
import json
from tqdm import tqdm
import argparse
import os
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--save_dir", default="pair_data", type=str)
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    legal_dict_json = args.legal_dict_json
    save_dir = args.save_dir
    
    
    print("Load legal dict json")
    doc_path = os.path.join(legal_dict_json)
    df = open(doc_path)
    doc_data = json.load(df)
    
    data = []
    
    for k, v in tqdm(doc_data.items()):
        save_dict = {}
        
        save_dict["docid"] = k
        save_dict["title"] = doc_data[k]["title"]
        save_dict["text"] = doc_data[k]["text"]
        
        data.append(save_dict)
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(os.path.join(save_dir, "corpus.jsonl"), "w") as f:
        json.dump(data, f)
        