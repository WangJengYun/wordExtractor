from sklearn.metrics.pairwise import cosine_similarity

def basic_similarity(
    doc_embedding, 
    word_embeddings, 
    candidates, 
    top_n):
    
    distances = cosine_similarity(doc_embedding, word_embeddings)[0]
    keywords = [
        (candidates[idx], round(distances[idx],4))
        for idx in distances.argsort()[::-1][:top_n]
    ]

    return keywords