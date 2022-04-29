from src import Entity_model, Flair_NER
from datasets import load_dataset, Dataset

import os
from sqlitedict import SqliteDict
from pathlib import Path
import argparse
from tqdm.auto import tqdm

def get_golden_context(x):
    set_golden_titles = set(x['supporting_facts']['title'])
    list_golden_pidx = [i for i, t in enumerate(x['context']['title']) if t in set_golden_titles]
    full_context = ""
    for i in list_golden_pidx:
        full_context += "".join(x['context']['sentences'][i])
        full_context += " "
    return full_context

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_framework", help="Flair, spacy")
    parser.add_argument("--spacy_model", default='en_core_web_sm')
    parser.add_argument("--flair_model", default='flair/ner-english-ontonotes-large')
    parser.add_argument("--flair_cache", default='./flair_cache')
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--output_dir", default='data/spacy')
    args = parser.parse_args()

    if args.ner_framework == 'spacy':
        ent_predictor = Entity_model(args.spacy_model)
    elif args.ner_framework == 'flair':
        ent_predictor = Flair_NER(args.flair_model, cache=args.flair_cache)
    else:
        raise ValueError('Unknown NER framework')

    dataset = load_dataset('hotpot_qa', 'distractor')
    

    for split in dataset.keys():
        if args.num_samples is not None:
            dataset_samples = Dataset.from_dict(dataset[split][:args.num_samples])
        else:
            dataset_samples = dataset[split]
        # 1) setup database
        output_dir = os.path.join(args.output_dir, 'hotpotqa', split)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        db_ent_question = SqliteDict(os.path.join(output_dir, 'question_ent.sqlite'))
        db_ent_context = SqliteDict(os.path.join(output_dir, 'context_ent.sqlite'))
        
        # 2) extract question entities
        dict_sent_idx2entities = ent_predictor.get_entities(dataset_samples['question'])
        for idx, list_ents in dict_sent_idx2entities.items():
            db_ent_question[str(idx)] = list_ents
        # 3) save question entities
        db_ent_question.commit()
        db_ent_question.close()

        # 4) extract context entities
        # for each QA instance
        for i, x in enumerate(tqdm(dataset_samples)):
            golden_context = get_golden_context(x)
            set_golden_titles = set(x['supporting_facts']['title'])
            dict_paragraph_idx2dict_sent_idx2entities = dict()
            char_offset = 0
            word_offset = 0
            # for each paragraph
            for p_idx, paragraph in enumerate(x['context']['sentences']):
                if x['context']['title'][p_idx] in set_golden_titles:
                    dict_sent_idx2entities = dict()
                    # for each setnence
                    for sent_idx, sent in enumerate(paragraph):
                        # extract entities
                        list_ents, spacy_sent = ent_predictor.get_entities_from_sentence(sent)
                        for e in list_ents:
                            # char idx
                            st = e['char_idx'][0] + char_offset
                            end = e['char_idx'][1] + char_offset
                            e['char_idx'] = (st, end)
                            # word idx
                            w_st = e['word_idx'][0] + word_offset
                            w_end = e['word_idx'][1] + word_offset
                            e['word_idx'] = (w_st, w_end)
                        dict_sent_idx2entities[sent_idx] = list_ents
                        char_offset += len(sent)
                        word_offset += len(spacy_sent)
                    dict_paragraph_idx2dict_sent_idx2entities[p_idx] = dict_sent_idx2entities
                    char_offset += 1 # for the space between paragraphs
            db_ent_context[i] = dict_paragraph_idx2dict_sent_idx2entities
        # 5) save context entities
        db_ent_context.commit()
        db_ent_context.close()
