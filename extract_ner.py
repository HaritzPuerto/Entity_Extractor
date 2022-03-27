from src import Entity_model, Flair_NER
from datasets import load_dataset, Dataset

import os
from sqlitedict import SqliteDict
from pathlib import Path
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset from HF")
    parser.add_argument("--ner_framework", help="Flair, spacy")
    parser.add_argument("--spacy_model", default='en_core_web_sm')
    parser.add_argument("--flair_model", default='flair/ner-english-ontonotes-large')
    parser.add_argument("--flair_cache", default='./flair_cache')
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--output_dir", default='data/ner')
    args = parser.parse_args()

    if args.ner_framework == 'spacy':
        ent_predictor = Entity_model(args.spacy_model)
    elif args.ner_framework == 'flair':
        ent_predictor = Flair_NER(args.flair_model, cache=args.flair_cache)
    else:
        raise ValueError('Unknown NER framework')

    dataset = load_dataset(args.dataset)
    

    for split in dataset.keys():
        if args.num_samples is not None:
            dataset = Dataset.from_dict(dataset[split][:args.num_samples])
        else:
            dataset = Dataset.from_dict(dataset[split])
        # 1) setup database
        output_dir = os.path.join(args.output_dir, args.dataset, split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        db_ent_question = SqliteDict(os.path.join(output_dir, 'question_ent.sqlite'))
        db_ent_context = SqliteDict(os.path.join(output_dir, 'context_ent.sqlite'))
        
        # 2) extract question entities
        dict_sent_idx2entities = ent_predictor.get_entities(dataset['question'])
        for idx, list_ents in dict_sent_idx2entities.items():
            db_ent_question[str(idx)] = list_ents
        # 3) save question entities
        db_ent_question.commit()
        db_ent_question.close()

        # 4) extract context entities
        dict_sent_idx2entities = ent_predictor.get_entities(dataset['context'])
        for idx, list_ents in dict_sent_idx2entities.items():
            db_ent_context[str(idx)] = list_ents 
        # 5) save context entities
        db_ent_context.commit()
        db_ent_context.close()
