from sklearn.metrics.pairwise import cosine_similarity

def basic_similarity(
    doc_embedding, 
    word_embeddings, 
    candidates,
    top_n,
    excluding_same_word):

    distances = cosine_similarity(doc_embedding, word_embeddings)[0]

    
    keywords = []
    for position in distances.argsort()[::-1]:
        candidate_name,candidate_info = candidates[position]
        
        if excluding_same_word:

            check_pattern = any([candidate_name in i for i, *_ in keywords])
            if check_pattern == False:
            
                keywords.append((candidate_name, candidate_info, round(distances[position],4)))
        
        else:
            keywords.append((candidate_name, candidate_info, round(distances[position],4)))
        

        if len(keywords) == top_n:
            break 
        
    return keywords