from src.graph_creation import SRL_model, Entity_model
from datasets import load_dataset

import os
import json
from pathlib import Path
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset from HF")
    parser.add_argument("--spacy_model", default='en_core_web_lg')
    args = parser.parse_args()

    ent_predictor = Entity_model(args.spacy_model)
    dataset = load_dataset(args.dataset)

    for split in args.dataset.keys():
        list_questions = [q.lstrip() for q in dataset['train']['question']]
        dict_ent_questions = ent_predictor.get_entities(list_questions)
        dict_ent_contexts = ent_predictor.get_entities(dataset['train']['context'])
        output_dir = os.path.join('data/ner/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, 'question_entities.json'), 'w') as f:
            json.dump(dict_ent_questions, f)
        with open(os.path.join(output_dir, 'context_entities.json'), 'w') as f:
            json.dump(dict_ent_contexts, f)
