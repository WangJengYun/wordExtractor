from backend import select_backend
from preprocess import Ckip_Transformers_Tokenizer, Keyword_Candidates
from similarity_algorithm import select_similarity

class KeyWordExtractor:
    def __init__(self, 
        embedding_model, 
        tokenizer_model,
        backend,
        stop_words = None,
        device = 'cuda'
        ):
        
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
        segment_by_stop_words = True,
        excluding_stop_words = True,
        similarity_method = 'basic',
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
                similarity_method = similarity_method, 
                top_n = top_n,
                min_df = min_df)
            
            return keywords
             
    def _extract_keywords_single_doc(
        self,
        doc,
        candidates,
        keyphrase_ngram_range,
        segment_by_stop_words,
        excluding_stop_words,
        similarity_method,
        top_n,
        min_df
        ):

        ws = self.tokenizer.tokenize(doc)[0]

        # Extract Candidates
        candidates = Keyword_Candidates(
                    stop_words = self.stop_words,
                    keyphrase_ngram_range = keyphrase_ngram_range,
                    segment_by_stop_words = segment_by_stop_words,
                    excluding_stop_words = excluding_stop_words,
                    min_df = min_df 
        ).extract_candidates(ws)
        
        # Extact Embedding 
        doc_embedding = self.model.doc_embed([ws])
        word_embeddings = self.model.word_embed(candidates, ws)

        # Calculate distances and extract keywords
        keywords = select_similarity(doc_embedding  = doc_embedding, 
                                     word_embeddings = word_embeddings, 
                                     candidates = candidates, 
                                     top_n = top_n,
                                     method = similarity_method)
        return keywords

if __name__ == '__main__':
    pass 
