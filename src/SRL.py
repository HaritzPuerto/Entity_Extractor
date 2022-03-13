from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.predictors.predictor import Predictor
from spacy.tokens import Doc

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

import torch.nn as nn
import torch.nn.functional as F


class SRL_model():
    def __init__(self, device='cuda', predictor_path=None):
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

    def get_srl_args(self, list_sentences):
        '''
        Input:
            - list_sentences
        Output:
            - dictionary that connecs sentence idx with its SRL arguments metadata
            - tokenized_batch: list of tokenized sentences (with input_ids, token_type_ids, attention_mask)
        '''
        
        # 3 - predict the SRL tags of each sentence
        dict_ins_idx2outputs, list_dict_token_idx2char_idx = self._srl_batch_predict(list_sentences)

        # 4 - process the SRL predictions
        dict_ins_idx2outputs = self._process_srl_predictions(list_sentences, dict_ins_idx2outputs, list_dict_token_idx2char_idx)
        return dict_ins_idx2outputs

    def _srl_batch_predict(self, list_tokenized_sentences):
        '''
        Extract SRL arguments in batch
        Input:
            - list_sentences
        Output:
            - dicionary with the srl arguments. Keys: instance index, values: 
        
        '''
        # create the batch    
        (batch_instances, dict_flattened_idx2sent_idx, list_dict_token_idx2char_idx) = self._create_batch_instances(list_tokenized_sentences)
        # in batch_instances all the sentences are flattened so we need the dict to get the map of flattened idx to the sent idx

        # predict the SRL arguments
        srl_predictions = self.predictor.predict_instances(batch_instances)
        # initialize the output dict
        dict_sent_idx2outputs = {sent_idx: [] for sent_idx in dict_flattened_idx2sent_idx.values()}
        # group the SRL predictions by sentence idx
        for flattened_idx, sent_idx in dict_flattened_idx2sent_idx.items():
            dict_sent_idx2outputs[sent_idx].append(srl_predictions['verbs'][flattened_idx])
        return dict_sent_idx2outputs, list_dict_token_idx2char_idx

    def _create_batch_instances(self, list_sentences):
        '''
        Input: list of tokenized sentences (i.e., list of words)
        Output:
            - list of spacy tokens
            - dictionary that maps each token of the first outupt to the index in the batch
        '''
        # based on predict_tokenized function from AllenNLP, it assumes spacy tokens
        # as we pretokenize the sentences, we need to follow the predict_tokenized function

        # 1 - create spacy tokens to make AllenNLP instances
        instances_num_sentXnumVerbs = []
        list_dict_token_idx2char_idx = []
        for sent in list_sentences:
            tokenized_sent = self._spacy_tokenizer.tokenize(sent)
            dict_token_idx2char_idx = {i: (x.idx, x.idx+len(x.text)) for i, x in enumerate(tokenized_sent)}
            list_dict_token_idx2char_idx.append(dict_token_idx2char_idx)

            instances_numVerbs = self.predictor.tokens_to_instances(tokenized_sent)
            instances_num_sentXnumVerbs.append(instances_numVerbs)
        
        # 3 - flatten the instances since each sentence is represented by a list of verbs
        i = 0
        flattened_instances = []
        dict_flattened_idx2sent_idx = dict()
        for sent_idx, instances_numVerbs in enumerate(instances_num_sentXnumVerbs):
            for ins in instances_numVerbs:
                flattened_instances.append(ins)
                dict_flattened_idx2sent_idx[i] = sent_idx
                i += 1

        return (flattened_instances, dict_flattened_idx2sent_idx, list_dict_token_idx2char_idx)
        
    def _process_srl_predictions(self, list_sentences, dict_ins_idx2outputs, list_dict_token_idx2char_idx):
        '''
        Input: 
            - dict_ins_idx2outputs: dictionary with the srl arguments. Keys: instance index, values:
            - tokenized_batch: transformer's tokenizer tokens (i.e., with ids, type_ids, attention_mask)
            - dict_sent_idx2token_idx2wordpiece_idx: dictionary that maps each token idx (whitespace tokens) to wordpiece idx (wordpiece tokens) # num_sent x num_tokens # 2 (i.e, starting and ending idx)
            - list_dict_token_idx2char_idx: maps sent idx to a dict that maps token idx to char idx
        Output:
            - list of SRL tuples # num_sent x num_verbs x num_args. Example of an SRL tuple:
                {'ARG0': {'word_idx': [0],
                        'wordpiece_idx': (1, 2),
                        'ids': [146],
                        'type_ids': [0],
                        'attention_mask': [1],
                        'tokens': ['I']},
                'V': {'word_idx': [2],
                        'wordpiece_idx': (3, 4),
                        'ids': [1684],
                        'type_ids': [0],
                        'attention_mask': [1],
                        'tokens': ['working']},
                'ARG1': {'word_idx': [3, 4, 5, 6, 7, 8],
                        'wordpiece_idx': (4, 11),
                        'ids': [1113, 170, 154, 1592, 1933, 1107, 1860],
                        'type_ids': [0, 0, 0, 0, 0, 0, 0],
                        'attention_mask': [1, 1, 1, 1, 1, 1, 1],
                        'tokens': ['on', 'a', 'Q', '##A', 'project', 'in', 'Germany']}},
        '''
        for i, list_srl_instances in dict_ins_idx2outputs.items():
            list_preds_instance = []
            for srl_instance in list_srl_instances:
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
                        st_char = list_dict_token_idx2char_idx[i][dict_tag2word_idx[tag]['word_idx'][0]][0] # first char of the first word
                        end_char = list_dict_token_idx2char_idx[i][dict_tag2word_idx[tag]['word_idx'][-1]][1] # last char of the last word
                        dict_tag2word_idx[tag]['char_idx'] = (st_char, end_char)
                        dict_tag2word_idx[tag]['text'] = list_sentences[i][st_char:end_char]
                    list_preds_instance.append(dict_tag2word_idx)
            yield list_preds_instance

