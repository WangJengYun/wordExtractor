from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings

from tokenization import Ckip_Transformers_Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

text = """
        監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
        [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
        [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
        監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
        最佳方案將使演算法能夠正確確定未見實例的類標籤。
        這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """
text_split = text.split('。')
WS = Ckip_Transformers_Tokenizer('./model_files/ckip_bert-base-chinese_ws/', use_device = 'cuda')


import re
from utils import read_dictionary 
stop_words = read_dictionary('./dict/stop_words.txt')
segment_char = read_dictionary('./dict/segment_char.txt')
segment_pattern = '|'.join([ re.escape(s) for s  in segment_char])
re.split(segment_pattern, doc)

result = WS.tokenize(sentences, stop_words=False)
result = [ ' '.join(sub_s) for sub_s in result]

segment_pattern_V1 = ' | '.join([ re.escape(s) for s  in stop_words])

input_list=  []
for sub_s in result:
    input_list.extend(re.split(segment_pattern_V1, sub_s))


doc = [' '.join(s) for s in result]

count = CountVectorizer(ngram_range=(1,2), stop_words = stop_words,token_pattern = r"(?u)\b\w\w+\b").fit(input_list)

candidates = count.get_feature_names()


count = CountVectorizer(ngram_range=(1,2), stop_words = stop_words,token_pattern = r"(?u)\b\w\w+\b").fit([AAA[0].replace('是','|')])


from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings

word_embed_model = TransformerWordEmbeddings('./model_files/ckip_bert-base-chinese')
doc_embed_model = DocumentPoolEmbeddings([word_embed_model])
doc_embed_model_A = TransformerDocumentEmbeddings('./model_files/ckip_bert-base-chinese')


doc = Sentence(candidates,use_tokenizer = False)
word_embed_model.embed(doc)


from backend import FlairEmbedder
emb_model = FlairEmbedder('./model_files/ckip_bert-base-chinese')

AA = emb_model.embed(candidates, type = 'word')