from src.graph_creation import SRL_model
from datasets import load_dataset

import os
import json
from pathlib import Path
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="from HF")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    srl_predictor = SRL_model(device=args.device)
    dataset = load_dataset(args.dataset)
    
    list_srl_questions = []
    list_srl_contexts = []
    for split in dataset.keys():
        dataset_len = len(dataset[split])
        for i in range(0, dataset_len, args.batch_size):
            # create batch of dataset instances
            j = i + args.batch_size
            list_questions = [q.lstrip() for q in dataset[split][i:j]['question']]
            # question
            list_srl_questions.extend(srl_predictor.get_srl_args(list_questions))
            # context
            list_contexts = dataset[split][i:j]['context']
            list_srl_contexts.extend(srl_predictor.get_srl_args(list_contexts))


        output_dir = os.path.join('data/srl/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, 'question_srl.json'), 'w') as f:
            json.dump({i: x for i, x in enumerate(list_srl_questions)}, f)
        with open(os.path.join(output_dir, 'context_srl.json'), 'w') as f:
            json.dump({i: x for i, x in enumerate(list_srl_contexts)}, f)
