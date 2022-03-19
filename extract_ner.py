from src import Entity_model
from datasets import load_dataset

import os
from sqlitedict import SqliteDict
from pathlib import Path
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset from HF")
    parser.add_argument("--spacy_model", default='en_core_web_sm')
    args = parser.parse_args()

    ent_predictor = Entity_model(args.spacy_model)
    dataset = load_dataset(args.dataset)

    for split in dataset.keys():
        # 1) setup database
        output_dir = os.path.join('data/ner/', args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        db_ent_question = SqliteDict(os.path.join(output_dir, 'question_ent.sqlite'))
        db_ent_context = SqliteDict(os.path.join(output_dir, 'context_ent.sqlite'))
        
        # 2) extract question entities
        dict_sent_idx2entities = ent_predictor.get_entities(dataset[split]['question'])
        for idx, list_ents in dict_sent_idx2entities.items():
            db_ent_question[str(idx)] = list_ents
        # 3) save question entities
        db_ent_question.commit()
        db_ent_question.close()

        # 4) extract context entities
        dict_sent_idx2entities = ent_predictor.get_entities(dataset[split]['context'])
        for idx, list_ents in dict_sent_idx2entities.items():
            db_ent_context[str(idx)] = list_ents 
        # 5) save context entities
        db_ent_context.commit()
        db_ent_context.close()
