from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.predictors.predictor import Predictor
from spacy.tokens import Doc

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

import torch.nn as nn
import torch.nn.functional as F
import spacy

class SRL_model():
    def __init__(self, device='cuda', predictor_path=None):
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self._spacy_tokenizer = SpacyTokenizer(language="en_core_web_sm", pos_tags=True)
        self.predictor = None
        if predictor_path is None:
            predictor_path = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        
        if device == 'cuda':
            self.predictor = Predictor.from_path(predictor_path, cuda_device=0)
        else:
            self.predictor = Predictor.from_path(predictor_path)

    def print_srl_tuple(self, srl_tuple):
        for arg_name, data in srl_tuple.items():
            arg_val = self.tokenizer.decode(data['ids'])
            print(f"{arg_name}: {arg_val}")
    
    def print_srl_tuples(self, list_srl_tuples):
        for i, srl_tuple in enumerate(list_srl_tuples):
            print("\t Tuple:", i)
            self.print_srl_tuple(srl_tuple)

    def get_srl_args(self, sent):
        sent = self.__clean_input(sent)
        tokenized_sent = self.spacy_nlp(sent)
        dict_token_idx2char_idx = {i: (x.idx, x.idx+len(x.text)) for i, x in enumerate(tokenized_sent)}
        srl_predictions = self.predictor.predict_tokenized([w.text for w in tokenized_sent])
        return self._process_srl_predictions(sent, srl_predictions, dict_token_idx2char_idx)

    def _process_srl_predictions(self, sent, list_srl_instances, dict_token_idx2char_idx):
        '''
        Input: 
            - list_sentensentces (str)
            - srl_predictions: dictionary with the srl arguments.
            - dict_token_idx2char_idx: maps word tokens to char idx
        Output:
            - list of SRL tuples num_verbs x num_args. Example of an SRL tuple:
                [{'ARG0': {'word_idx': [0, 1, 2, 3, 4],
                           'char_idx': (0, 28),
                           'text': 'What institute at Notre Dame'},
                'V': {'word_idx': [5], 'char_idx': (29, 36), 'text': 'studies'},
                'ARG1': {'word_idx': [6, 7, 8, 9, 10],
                         'char_idx': (37, 69),
                         'text': 'the reasons for violent conflict'}}
                ]
        '''
        list_preds_instance = []
        for srl_instance in list_srl_instances['verbs']:
            # process the output of the SRL model to create a list of SRL tags
            # each element of the list is a dictionary SRL_tag -> {'word_idx': ..., 'wordpiece_idx': (st, end)}
            dict_tag2word_idx = dict()
            # for each SRL tag
            for word_idx, tag in enumerate(srl_instance['tags']):
                # skip non-SRL tags
                if tag != 'O':
                    tag = tag.split("-")[-1]
                    # and SRL arg may have more than one word
                    if tag in dict_tag2word_idx:
                        dict_tag2word_idx[tag]['word_idx'].append(word_idx)
                    else:
                        dict_tag2word_idx[tag] = {'word_idx': [word_idx]} 
            if len(dict_tag2word_idx) > 1:
                # we have stored a list of pairs of start and end wordpiece idx for each word
                # we need to convert it into a pair of (st, end)
                for tag in dict_tag2word_idx.keys():
                    st_char = dict_token_idx2char_idx[dict_tag2word_idx[tag]['word_idx'][0]][0] # first char of the first word
                    end_char = dict_token_idx2char_idx[dict_tag2word_idx[tag]['word_idx'][-1]][1] # last char of the last word
                    dict_tag2word_idx[tag]['char_idx'] = (st_char, end_char)
                    dict_tag2word_idx[tag]['text'] = sent[st_char:end_char]
                list_preds_instance.append(dict_tag2word_idx)
        return list_preds_instance

    def __clean_input(self, s):
        '''
        Remove duplicated whitespaces.
        There are several instances in SQuAD with double spaces that makes tricky the allignment in the tokenization.
        '''
        return " ".join(s.split())