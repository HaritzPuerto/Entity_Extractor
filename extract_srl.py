from src import SRL_model
from datasets import load_dataset
from tqdm import trange
from sqlitedict import SqliteDict

import os
import json
from pathlib import Path
import argparse


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
        # 1) setup database
        output_dir = os.path.join('data/srl/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        db_srl_questions = SqliteDict(os.path.join(output_dir, 'question_srl.sqlite'))
        db_question_errors = SqliteDict(os.path.join(output_dir, 'question_errors.sqlite'))
        db_srl_contexts = SqliteDict(os.path.join(output_dir, 'context_srl.sqlite'))
        db_context_errors = SqliteDict(os.path.join(output_dir, 'context_errors.sqlite'))


        dataset_len = len(dataset[split])
        for i in trange(0, dataset_len, args.batch_size):
            # create batch of dataset instances
            j = i + args.batch_size
            list_questions = dataset[split][i:j]['question']
            # question
            try:
                srl_pred = list(srl_predictor.get_srl_args(list_questions))
                for idx in range(i,j):
                    db_srl_questions[str(idx)] = srl_pred[idx-i]
                    dict_srl_questions[idx] = srl_pred[idx-i]
            except:
                for idx in range(i,j):
                    db_srl_questions[str(idx)] = []
                    db_question_errors[str(idx)] = list_questions[idx-i]
            
            # context
            list_contexts = dataset[split][i:j]['context']
            try:
                srl_pred = list(srl_predictor.get_srl_args(list_contexts))
                for idx in range(i,j):
                    db_srl_contexts[str(idx)] = srl_pred[idx-i]
                    
            except:
                for idx in range(i,j):
                    db_srl_contexts[str(idx)] = []
                    db_context_errors[str(idx)] = list_contexts[idx-i]
            
        db_srl_questions.commit()   
        db_srl_contexts.commit()
        db_srl_questions.close()
        db_srl_contexts.close()

        db_question_errors.commit()
        db_context_errors.commit()
        db_question_errors.close()
        db_context_errors.close()