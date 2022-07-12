
def select_similarity(doc_embedding, 
                      word_embeddings, 
                      candidates, 
                      method,
                      top_n,
                      excluding_same_word):

    if method == 'basic':
        
        from .basic_similarity import basic_similarity
        return basic_similarity(doc_embedding, 
                                word_embeddings, 
                                candidates,
                                top_n,
                                excluding_same_word)