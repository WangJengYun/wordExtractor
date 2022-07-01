
def select_backend(embedding_model, backend):
    
    if backend == 'flair':
        
        from .flair_Embedder import FlairEmbedder
        return FlairEmbedder(embedding_model)

