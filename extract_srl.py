from src import SRL_model
from datasets import load_dataset
from tqdm.auto import tqdm
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
    
    for split in dataset.keys():
        # 1) setup database
        output_dir = os.path.join('data/srl/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        db_srl_questions = SqliteDict(os.path.join(output_dir, 'question_srl.sqlite'))
        db_srl_contexts = SqliteDict(os.path.join(output_dir, 'context_srl.sqlite'))
        db_errors = SqliteDict(os.path.join(output_dir, 'errors.sqlite'))

        # 2) extract SRL
        for i, x in enumerate(tqdm(dataset[split])):
            try:
                # question
                db_srl_questions[str(i)] = srl_predictor.get_srl_args(x['question'])
                # context
                db_srl_contexts[str(i)] = srl_predictor.get_srl_args(x['context'])
            except:
                db_errors[str(i)] = i
        
        # 3) save to DB
        db_srl_questions.commit()   
        db_srl_contexts.commit()
        db_srl_questions.close()
        db_srl_contexts.close()

        db_errors.commit()
        db_errors.close()
