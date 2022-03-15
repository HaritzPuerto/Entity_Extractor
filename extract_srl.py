from src import SRL_model
from datasets import load_dataset
from tqdm import trange

import os
import json
from pathlib import Path
import argparse


def clean_input(s):
    return " ".join(s.lstrip().rstrip().split())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="from HF")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path", default='models/structured-prediction-srl-bert.2020.12.15.tar.gz')
    args = parser.parse_args()

    srl_predictor = SRL_model(device=args.device, predictor_path=args.model_path)
    dataset = load_dataset(args.dataset)
    
    dict_srl_questions = dict()
    dict_srl_contexts = dict()
    list_errors = []
    for split in dataset.keys():
        dataset_len = len(dataset[split])
        if split == 'train':
            dataset_len = 1000 
        for i in trange(0, dataset_len, args.batch_size):
            # create batch of dataset instances
            j = i + args.batch_size
            list_questions = [clean_input(q) for q in dataset[split][i:j]['question']]
            # question
            try:
                srl_pred = list(srl_predictor.get_srl_args(list_questions))
                for idx in range(i,j):
                    dict_srl_questions[idx] = srl_pred[idx-i]
            except:
                for idx in range(i,j):
                    dict_srl_questions[idx] = []]
                list_errors.append((i, list_questions))
                with open('./errors.json', 'w') as f:
                    json.dump(list_errors, f)
            
            # context
            list_contexts = [clean_input(x) for x in dataset[split][i:j]['context']]
            try:
                srl_pred = list(srl_predictor.get_srl_args(list_contexts))
                for idx in range(i,j):
                    dict_srl_contexts[idx] = srl_pred[idx-i]
            except:
                for idx in range(i,j):
                    dict_srl_contexts[idx] = []
                list_errors.append((i, list_contexts))
                with open('./errors.json', 'w') as f:
                    json.dump(list_errors, f)
            
            
        output_dir = os.path.join('data/srl/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, 'question_srl.json'), 'w') as f:
            json.dump(dict_srl_questions, f)
        with open(os.path.join(output_dir, 'context_srl.json'), 'w') as f:
            json.dump(dict_srl_contexts, f)
