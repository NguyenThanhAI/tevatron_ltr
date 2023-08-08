from typing import List, Dict

import numpy as np
import json
from tqdm import tqdm
import argparse
import os
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--test_json_path", type=str, default="for_test_question_answer.json")
    parser.add_argument("--save_dir", default="pair_data", type=str)
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    test_json_path = args.test_json_path
    save_dir = args.save_dir
    
    train_path = os.path.join(data_dir, test_json_path)
    data = json.load(open(train_path))
    items = data["items"]
    print("Num test question: {}".format(len(items)))
    
    test_data = []
    
    for idx, item in tqdm(enumerate(items)):
        query_data = {}
        
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        
        query_data["query_id"] = question_id
        query_data["query"] = question
        
        query_data["positive_passages"] = []
        
        query_data["negative_passages"] = []
        
        
        test_data.append(query_data)
        
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dev.jsonl"), "w") as f:
        json.dump(test_data, f)