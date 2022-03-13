import spacy
from tqdm import tqdm

class Entity_model():
    def __init__(self, spacy_model='en_core_web_sm'):
        self.predictor = None
        self.nlp = spacy.load(spacy_model)


    def get_entities(self, list_sentences):
        '''
        Input:
            - list_sentences
        Output:
            - dict_sent_idx2entities: dictionary that maps each sentence idx to a list of entities
        '''
        dict_sent_idx2entities = dict() # this will be the final output
        # for each sentence
        for sent_idx, sent in enumerate(tqdm(list_sentences)):
            # initialize the list of entities for the current sentence
            dict_sent_idx2entities[sent_idx] = [] 
            # use spacy processor on the question | context sentence
            spacy_sent = self.nlp(sent) 
            # get the entities of the two sentences concatenated (query and context)
            for e in spacy_sent.ents: 
                # create the dictionary to store the metadata of the entity
                dict_ent = {'char_idx': (e.start_char, e.end_char), # eg: [0,1,2]
                            'ent_type': e.label_, # eg: "PERSON"
                            'text': e.text, # eg: "John"
                            }
                dict_sent_idx2entities[sent_idx].append(dict_ent)
        return dict_sent_idx2entities