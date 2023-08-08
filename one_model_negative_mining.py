import os
import argparse
import json
import pickle
from tqdm import tqdm

import numpy as np
import shutil

import torch

from sentence_transformers import util

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="phobert_base")
    parser.add_argument("--round_turn", type=int, default=1)
    parser.add_argument("--data_path", default="zac2021-ltr-data", type=str, help="path to input data")
    parser.add_argument("--save_dir", default="pair_data", type=str)
    parser.add_argument("--top_k", default=20, type=int, help="top k hard negative mining")
    parser.add_argument("--path_doc_refer", default="generated_data/doc_refers_saved.pkl", type=str, help="path to doc refers")
    parser.add_argument("--path_legal", default="generated_data/legal_dict.json", type=str, help="path to legal dict")
    parser.add_argument("--encoded_data_dir", type=str)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    
    args = get_args()
    
    
    model_name = args.model_name
    round_turn = args.round_turn
    data_path =args.data_path
    save_dir =args.save_dir
    top_k =args.top_k
    path_doc_refer = args.path_doc_refer
    path_legal =args.path_legal
    encoded_data_dir = args.encoded_data_dir
    corpus_embedding_path = os.path.join(encoded_data_dir, "{}_tevatron_dpr_round_{}".format(model_name, round_turn), "{}_tevatron_dpr_round_{}_corpus_emb.pkl".format(model_name, round_turn))
    train_question_embedding_path = os.path.join(encoded_data_dir, "{}_tevatron_dpr_round_{}".format(model_name, round_turn), "{}_tevatron_dpr_round_{}_train_question_emb.pkl".format(model_name, round_turn))
    
    
    # load training data from json
    data = json.load(open(os.path.join(data_path, "for_train_question_answer.json")))
    
    items = data["items"]
    print(len(items))
    
    with open(path_doc_refer, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_path = os.path.join(path_legal)
    df = open(doc_path)
    doc_data = json.load(df)
    
    with open(corpus_embedding_path, "rb") as f:
        corpus_embedding = pickle.load(f)

    with open(train_question_embedding_path, "rb") as f:
        train_question_embedding = pickle.load(f)
    
    print("Check valid index and id")
    for idx, (k, v) in tqdm(enumerate(doc_data.items())):
        assert k == corpus_embedding[1][idx]
        assert corpus_embedding[1][idx] == (doc_refers[idx][0] + "_" + doc_refers[idx][1])
    
    for idx, item in tqdm(enumerate(items)):
        assert train_question_embedding[1][idx] == item["question_id"]
        
    matrix_emb = corpus_embedding[0].astype(np.float32)
    question_emb_dict = train_question_embedding[0].astype(np.float32)
    
    train_data = []
    
    for idx, item in tqdm(enumerate(items)):
        query_data = {}
        
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        
        query_data["query_id"] = question_id
        query_data["query"] = question

        query_data["positive_passages"] = []
        
        for article in relevant_articles:
            save_dict = {}
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["docid"] = concat_id
            save_dict["title"] = doc_data[concat_id]["title"]
            save_dict["text"] = doc_data[concat_id]["text"]

            query_data["positive_passages"].append(save_dict)
            
        encoded_question = question_emb_dict[idx]
        
        all_cosine = util.cos_sim(torch.from_numpy(encoded_question), torch.from_numpy(matrix_emb)).numpy().squeeze(0)
        new_scores = all_cosine
        predictions = np.argsort(new_scores)[::-1][:top_k]
        
        query_data["negative_passages"] = []

        for idx_2, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]

            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
                    break

            if check == 0:
                save_dict = {}
                concat_id = pred[0] + "_" + pred[1]
                save_dict["docid"] = concat_id
                save_dict["title"] = doc_data[concat_id]["title"]
                save_dict["text"] = doc_data[concat_id]["text"]
                query_data["negative_passages"].append(save_dict)

        train_data.append(query_data)
    
    save_dir = os.path.join(save_dir, model_name, "{}_tevatron_dpr_round_{}".format(model_name, round_turn), "train_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(os.path.join(save_dir, "train.jsonl"), "w") as f:
        json.dump(train_data, f)