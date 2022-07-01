from backend.base import BaseEmbedder
import numpy as np 
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings


class FlairEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        if isinstance(embedding_model, str):
            
            self.word_embedding_model = TransformerWordEmbeddings(embedding_model)
            self.doc_embedding_model = DocumentPoolEmbeddings([self.word_embedding_model])

    def doc_embed(self, text, single = True):
        embeddings = None

        if single:
            sentence = Sentence(text) 
            self.doc_embedding_model.embed(sentence)
            embeddings = sentence.embedding.detach().cpu().numpy()
        
        else:
            embeddings = []
            if type == 'doc':
                for idx , doc in enumerate(text):

                    sentence = Sentence(doc) 
                    
                    self.doc_embedding_model.embed(sentence)
                    embeddings.append(sentence.embedding.detach().cpu().numpy())

                embeddings = np.asarray(embeddings)
            elif type == 'word' :
                sentence = Sentence(text) 
                self.word_embedding_model.embed(sentence)
                assert len(sentence) == len(text)
                for idx in range(len(sentence)):
                    embeddings.append(sentence[idx].embedding.detach().cpu().numpy())

                embeddings = np.asarray(embeddings)
            else:
                ValueError('Please check value of type')

        return embeddings