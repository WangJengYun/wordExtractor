import re 
import numpy as np 
from utils import read_dictionary 
from tokenization import Ckip_Transformers_Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from backend import select_backend

class KeyWordExtractor:
    def __init__(self, 
        embedding_model, 
        backend, 
        tokenizer_model,
        stop_words = None,  
        device = 'cuda'):
        
        self.stop_words = stop_words
        self.device = device
        
        self.tokenizer = Ckip_Transformers_Tokenizer(tokenizer_model,\
                                                     use_device = self.device)
        
        self.model = select_backend(embedding_model, backend)                                           

    def extract_keywords(
        self,
        docs,
        candidates = None,
        keyphrase_ngram_range = (1, 1),
        segment_by_stop_words = False,
        excluding_stop_words = True,
        top_n = 5,
        min_df = 1
        ):
        if isinstance(docs, str):
            keywords = self._extract_keywords_single_doc(
                doc = docs,
                candidates = candidates,
                keyphrase_ngram_range = keyphrase_ngram_range,
                segment_by_stop_words = segment_by_stop_words,
                excluding_stop_words = excluding_stop_words,
                top_n = top_n,
                min_df = min_df)
            
            return keywords
    
    def _tokenize(self, 
        ws,
        segment_by_stop_words = False
        ):
        ws = [s.strip() for s in ws if s.strip() != '']
        if segment_by_stop_words:
            sentences = self._split_sentence_by_stop_words(ws)
        else:
            sentences = ' '.join(ws)
        return sentences

    def _split_sentence_by_stop_words(self, ws):
        split_sentences = None 
        if self.stop_words :
            segment_stop_words_pattern = ' | '.join([ re.escape(s) for s  in self.stop_words])
            split_sentences = re.split(segment_stop_words_pattern, ' '.join(ws))
        else:
            split_sentences = ' '.join(ws)
       
        return split_sentences
         
    def _extract_keywords_single_doc(
        self,
        doc,
        candidates = None,
        keyphrase_ngram_range = (1, 1),
        segment_by_stop_words = False,
        excluding_stop_words = True,
        top_n = 5,
        min_df = 1
        ):
        ws = self.tokenizer.tokenize(doc)[0]
        sentences_for_candidates = self._tokenize(ws, segment_by_stop_words = segment_by_stop_words)
        
        if candidates is None:
            if excluding_stop_words:
                assert self.stop_words is not None
                input_stop_words = self.stop_words
            else:
                input_stop_words = None 

            count = CountVectorizer(
                ngram_range = keyphrase_ngram_range, stop_words = input_stop_words, min_df = min_df
                ).fit(sentences_for_candidates)

            candidates = count.get_feature_names()
        candidates, candidates_positions = self._get_words_positions(ws, candidates)
        
        # Extact Embedding 
        doc_embedding = self.model.doc_embed([ws])
        word_embedding = self.model.word_embed(zip(candidates,candidates_positions), ws)


        distances = cosine_similarity(doc_embedding, word_embedding)[0]
        keywords = [
            (candidates[idx], round(distances[idx],4))
            for idx in distances.argsort()[::-1][:top_n]
        ]
        return keywords

    def _get_words_positions(self, ws, candidates):
        ws = np.array(ws)
        words = []
        words_positions = []
        for phrase in candidates:    
            word_list = phrase.split(' ')
            n_split = len(word_list)
            n = 0
            positions = None  
            while n < n_split:
                if n == 0:
                    positions = np.where(ws == word_list[n])[0]
                else:
                    positions = positions[ws[positions + n] == word_list[n]]

                n += 1 
            if positions.tolist():
                words_positions.append((positions.tolist(), n_split))
                words.append(phrase)
        return words, words_positions

if __name__ == '__main__':
    pass 
