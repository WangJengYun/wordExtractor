from .base import BaseEmbedder
import numpy as np 
import flair, torch
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings

flair.device = torch.device('cpu') 

class FlairEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        if isinstance(embedding_model, str):
            
            self.word_embedding_model = TransformerWordEmbeddings(embedding_model)
            self.doc_embedding_model = DocumentPoolEmbeddings([self.word_embedding_model])

    def doc_embed(self, texts):
        try:
            embeddings = []
            for doc in texts:
                sentence = Sentence(doc) 
                self.doc_embedding_model.embed(sentence)
                embeddings.append(sentence.embedding.detach().cpu().numpy())

            embeddings = np.asarray(embeddings)
        except:
            print(doc)
            print(sentence)
            raise ValueError

        return embeddings
    #  doc_split, words =  ws, candidates
    def word_embed(self, words, doc_split = None):
        embeddings = None 
        if doc_split is None:
            embeddings = None 
        else :
            doc = Sentence(doc_split)
            self.word_embedding_model.embed(doc)
            
            all_words_embedding = []
            for word_name, word_position in words:
                # word_name, word_position = words[7]
                position, n_split, *_ = word_position
                position_list = [[ i +j for j in range(n_split)] for i in position]
                word_embedding = []
                for gram_position in position_list:
                    gram_embeddings = [
                        doc[token_id].embedding.detach().cpu().numpy() for token_id in  gram_position
                        ]
                    word_embedding.append(np.stack(gram_embeddings, axis = 0).mean(axis = 0))

                all_words_embedding.append(np.stack(word_embedding,axis = 0).mean(axis = 0))

            embeddings = np.stack(all_words_embedding,axis = 0)

        return embeddings