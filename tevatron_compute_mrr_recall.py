from typing import List, Dict

import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer, calculate_f2

from sentence_transformers import util
from utils import compute_metrics


def load_bm25(bm25_path) -> BM25Plus:
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25


def load_encoded_legal_data(encoded_legal_path):
    print("Start loading embedding of legal data")
    with open(encoded_legal_path, "rb") as f:
        emb_legal_data = pickle.load(f)
    return emb_legal_data[0].astype(np.float32)


def load_encoded_question_data(question_encoded_path):
    print("Start loading embedding of question data")
    with open(question_encoded_path, "rb") as f:
        emb_question_data = pickle.load(f)
        
    return emb_question_data[0].astype(np.float32)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="zac2021_ltr_data")
    parser.add_argument("--test_json_path", type=str, default="for_test_question_answer.json")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--encoded_data_dir", type=str)
    parser.add_argument("--range_score", default=2.6, type=float)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    test_json_path = args.test_json_path
    legal_dict_json = args.legal_dict_json
    bm25_path = args.bm25_path
    doc_refers_path = args.doc_refers_path
    encoded_data_dir = args.encoded_data_dir
    range_score = args.range_score
    
    test_path = os.path.join(data_dir, test_json_path)
    data = json.load(open(test_path))
    items = data["items"]
    print("Num test question: {}".format(len(items)))
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path=bm25_path)
    
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
    
    model_names = []
    
    emb_legal_data: Dict[str, np.ndarray] = {}
    question_emb_dict: Dict[str, np.ndarray] = {}
    
    print("Load embedding of corpus and test query questions")
    for model_name in tqdm(model_names):
        encoded_legal_path = os.path.join(encoded_data_dir, model_name, "{}_corpus_emb.pkl".format(model_name))
        question_encoded_path = os.path.join(encoded_data_dir, model_name, "{}_test_question_emb.pkl".format(model_name))
        emb_legal_data[model_name] = load_encoded_legal_data(encoded_legal_path=encoded_legal_path)
        question_emb_dict[model_name] = load_encoded_question_data(question_encoded_path=question_encoded_path)
        
    top_n = 100
    
    weights = [0.25, 0.25, 0.25, 0.25]
    
    mrr_total = 0
    r1_total = 0
    r5_total = 0
    r10_total = 0
    r20_total = 0
    r25_total = 0
    r50_total = 0
    
    for idx, item in tqdm(enumerate(items)):
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        cos_sim = []
        
        for idx_2, model_name in enumerate(model_names):
            emb1 = question_emb_dict[model_name][idx]
            emb2 = emb_legal_data[model_name]
            
            scores = util.cos_sim(torch.from_numpy(emb1), torch.from_numpy(emb2))
            
            cos_sim.append(weights[idx_2] * scores)
            
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        new_scores = doc_scores * cos_sim
        max_score = np.max(new_scores)
        
        predictions = np.argsort(new_scores)[::-1][:top_n]
        
        
        mrr_el, r1_el, r5_el, r10_el, r20_el, r25_el, r50_el = compute_metrics(predictions=predictions,
                                                                              doc_refers=doc_refers,
                                                                              relevant_docs=relevant_articles)
        
        
        mrr_total += mrr_el
        r1_total += r1_el
        r5_total += r5_el
        r10_total += r10_el
        r20_total += r20_el
        r25_total += r25_el
        r50_total += r50_el
        
        print("MRR: {:.3f}, R@1: {:.3f}, R@5: {:.3f}, R@10: {:.3f}, R@20: {:.3f}, R@25: {:.3f}, R@50: {:.3f}".format(mrr_total/(idx + 1), (r1_total / (idx + 1)), (r5_total / (idx + 1)), (r10_total / (idx + 1)), (r20_total / (idx + 1)), (r25_total / (idx + 1)), (r50_total / (idx + 1))))
        
    print("MRR: {:.3f}\n R@1: {:.3f}\n R@5: {:.3f}\n R@10: {:.3f}\n R@20: {:.3f}\n R@25: {:.3f}\n R@50: {:.3f}".format(mrr_total/len(items), (r1_total / len(items)), (r5_total / len(items)), (r10_total / len(items)), (r20_total / len(items)), (r25_total / len(items)), (r50_total / len(items))))