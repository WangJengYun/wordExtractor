import re 
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
        segment_char = None,
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
                segment_char = segment_char,
                segment_by_stop_words = segment_by_stop_words,
                excluding_stop_words = excluding_stop_words,
                top_n = top_n,
                min_df = min_df)
            
            return keywords
    
    def _tokenize(self, 
        ws,
        segment_by_stop_words = False
        ):

        if isinstance(docs, str):
            ws = [sub_s.strip() for sub_s in ws if  sub_s.strip() != '']
            if segment_by_stop_words:
                sentences = self._split_sentence_by_stop_words(ws)
            else:
                sentences = ' '.join(ws)
            return sentences

        # elif isinstance(docs, list):
        #     docs_sentences = []
        #     for d in docs:
        #         sentences = self._split_doc(d , segment_char)
        #         sentences = [' '.join(sub_s) for sub_s in self.tokenizer.tokenize(sentences)]
        #         if segment_by_stop_words:
        #             sentences = self._split_sentence_by_stop_words(sentences)
        #         docs_sentences.append(sentences)

        #     return docs_sentences

    def _split_doc(self, doc, segment_char):
        if segment_char :
            segment_char_pattern = '|'.join([ re.escape(s) for s  in segment_char])
            sentences = re.split(segment_char_pattern, doc)
        else:
            sentences = [doc]
        
        sentences = [ s.strip() for s in sentences if s.strip() != '']

        return sentences

    def _split_sentence_by_stop_words(self, ws):
        split_sentences = None 
        if self.stop_words :
            segment_stop_words_pattern = ' | '.join([ re.escape(s) for s  in self.stop_words])
            split_sentences = re.split(segment_stop_words_pattern, ' '.join(ws)))
        else:
            split_sentences = ' '.join(ws)
       
        return split_sentences
         
    def _extract_keywords_single_doc(
        self,
        doc,
        candidates = None,
        keyphrase_ngram_range = (1, 1),
        segment_char = None,
        segment_by_stop_words = False,
        excluding_stop_words = True,
        top_n = 5,
        min_df = 1
        ):
        sentences = self.tokenizer.tokenize(doc)
        
        sentences = self._tokenize(doc, segment_char = segment_char, segment_by_stop_words = segment_by_stop_words)
        if candidates is None:
            if excluding_stop_words:
                assert self.stop_words is not None
                input_stop_words = self.stop_words
            else:
                input_stop_words = None 

            count = CountVectorizer(
                ngram_range = keyphrase_ngram_range, stop_words = input_stop_words, min_df = min_df
                ).fit(sentences)

            candidates = count.get_feature_names()
        
        # # Extact Embedding 
        # doc_embedding = self.model.embed([self.tokenizer.tokenize(doc)], type = 'doc')
        # word_embedding = self.model.embed(candidates, type = 'word')
# 
# 
        # distances = cosine_similarity(doc_embedding, word_embedding)[0]
        # keywords = [
        #     (candidates[idx], round(distances[idx],4))
        #     for idx in distances.argsort()[::-1][:top_n]
        # ]
        return candidates

if __name__ == '__main__':
    pass 
