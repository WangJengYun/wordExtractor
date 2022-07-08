from utils import read_dictionary 
from model import KeyWordExtractor

# from KeyExtractor.core import KeyExtractor

stop_words = read_dictionary('./dict/stop_words.txt')
segment_char = read_dictionary('./dict/segment_char.txt')
   
doc = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範s例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
"""
   
kw_model = KeyWordExtractor(embedding_model = './model_files/ckip_bert-base-chinese', 
                            tokenizer_model ='./model_files/ckip_bert-base-chinese_ws/',
                            backend = 'flair',
                            stop_words = stop_words,
                            device = 'cuda')
   
keywords = kw_model.extract_keywords(doc, 
                                    keyphrase_ngram_range = (1, 7),
                                    segment_by_stop_words = True,
                                    min_df = 2,
                                    similarity_method = 'basic',
                                    top_n = 10)

























#------------------------------------------
model = FlairEmbedder('./model_files/ckip_bert-base-chinese')

word_embedding_2 = model.word_embed(candidates, ws)
doc_embedding = model.doc_embed([ws])

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(doc_embedding[0],word_embedding[7])

cosine_similarity(doc_embedding,np.asarray([all_words_embedding[7]]))
cosine_similarity(doc_embedding,np.asarray([word_embedding_2[7]]))

distances = cosine_similarity(doc_embedding, word_embedding_2)[0]
keywords = [
            (candidates[idx], round(distances[idx],4))
            for idx in distances.argsort()[::-1][:100]
        ]
#================================================
from KeyExtractor.utils import tokenization as tk
from model import KeyWordExtractor
tokenizer = tk.TokenizerFactory(name="ckip-transformers-albert-tiny")
tokenized_text  = tokenizer.tokenize(doc)[0]

from KeyExtractor.core import KeyExtractor
ke = KeyExtractor(embedding_method_or_model="ckiplab/bert-base-chinese")
keywords = ke.extract_keywords(tokenized_text, n_gram=1, top_n=5)

ke._get_doc_embeddings(doc)

_, n_gram_text = ke._preprocess(tokenized_text, n_gram = 1)
results = ke._evaluate(tokenized_text, n_gram_text)
ret = ke._postprocess(results, 10)

doc_embeddings = ke._get_doc_embeddings(tokenized_text)


from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings

word_embed_model = TransformerWordEmbeddings('./model_files/ckip_bert-base-chinese')
doc_embed_model = DocumentPoolEmbeddings([word_embed_model])

doc = Sentence(ws)
doc_embed_model.embed(doc)

doc = Sentence(tokenized_text)
word_embed_model.embed(doc)


AAA = kw_model.tokenizer.tokenize(doc)
model = FlairEmbedder('./model_files/ckip_bert-base-chinese')
model.embed(tokenized_text,type = 'doc')

from sklearn.metrics.pairwise import cosine_similarity
cosineSimilarity(word[2].embedding, torch.tensor(doc_embeddingss[0]).to('cuda'))
cosineSimilarity(word[2].embedding, doc_embeddings)
cosineSimilarity(word[5].embedding, doc_embeddings)

word[2].embedding
results[2].embeddings


word = Sentence('學習')
word_embed_model.embed(word)
word[0].embedding
cosineSimilarity(word[0].embedding, torch.tensor(doc_embeddingss[0]).to('cuda'))
cosineSimilarity(word[0].embedding, doc_embeddings)


cosineSimilarity(torch.tensor(word_embedding[7]).to('cuda'), doc.embedding)