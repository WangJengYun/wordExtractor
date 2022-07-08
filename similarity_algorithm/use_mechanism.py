
def select_similarity(doc_embedding, 
                      word_embeddings, 
                      candidates, 
                      top_n,
                      method):
                      
    candidates = [w[0] for w in candidates]
    if method == 'basic':
        
        from .basic_similarity import basic_similarity
        return basic_similarity(doc_embedding, 
                                word_embeddings, 
                                candidates, 
                                top_n)