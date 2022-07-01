class BaseEmbedder:
    def __init__(self, word_embedding_model = None, doc_embedding_model = None):

        self.word_embedding_model = word_embedding_model
        self.doc_embedding_model = doc_embedding_model

    def doc_embed(self, text):
        pass 
    
    def embed(self, text, type):
        pass 

