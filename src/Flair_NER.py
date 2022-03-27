import spacy
from tqdm.auto import tqdm
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from pathlib import Path



class Flair_NER():
    def __init__(self, flair_model='flair/ner-english-ontonotes-large', cache='./flair_cache'):
        flair.cache_root = Path(cache)
        self.tagger = SequenceTagger.load(flair_model)
        self.spacy_nlp = spacy.load('en_core_web_sm')

    def __clean_input(self, s):
        '''
        Remove duplicated whitespaces.
        There are several instances in SQuAD with double spaces that makes tricky the allignment in the tokenization.
        '''
        return " ".join(s.split())

    def get_entities(self, list_docs):
        '''
        Input:
            - list_sentences
        Output:
            - dict_sent_idx2entities: dictionary that maps each sentence idx to a list of entities
        '''
        dict_doc_idx2entities = dict()
        # for each doc
        for doc_idx, doc in enumerate(tqdm(list_docs)):
            dict_doc_idx2entities[doc_idx] = []
            spacy_doc = self.spacy_nlp(self.__clean_input(doc))
            # for each sentence
            for spacy_sent in spacy_doc.sents:
                sentence = Sentence([w.text for w in spacy_sent])
                # predict NER tags
                self.tagger.predict(sentence)
                # for each ner
                for e in sentence.get_spans('ner'):
                    sent_word_st = e.tokens[0].idx-1
                    sent_word_end = e.tokens[-1].idx-1
                    
                    doc_word_st = spacy_sent[sent_word_st].i
                    doc_word_end = spacy_sent[sent_word_end].i+1
                    
                    char_st = spacy_sent[sent_word_st].idx
                    char_end = spacy_sent[sent_word_end].idx + len(spacy_sent[sent_word_end].text)
                    
                    dict_ent = {'char_idx': (char_st, char_end), # eg: (0,3)
                                'word_idx': (doc_word_st, doc_word_end), # eg: (0,1)
                                'ent_type': e.labels[0].value, # eg: "PERSON"
                                'text': spacy_doc.text[char_st:char_end], # eg: "John"
                                }
                    # assert spacy_doc[doc_word_st:doc_word_end].text == e.text
                    # assert spacy_doc.text[char_st:char_end] == e.text
                    
                    dict_doc_idx2entities[doc_idx].append(dict_ent)
        return dict_doc_idx2entities