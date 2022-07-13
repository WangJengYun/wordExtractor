from re import S
from tqdm import  tqdm
from functools import partial
from multiprocessing import Manager, Pool
from similarity_algorithm import select_similarity

class Parallel_Computing(object):
    def __init__(self, n_process):

        self.pool = Pool(n_process)
        self.value_collection = Manager().list()
    
    def run(self, func, inputs, default_args = {}):
        
        if 'value_collection' in func.__code__.co_varnames:
            default_args['value_collection'] = self.value_collection        
        
        if default_args:
            func = partial(func, **default_args)

        self.pool.starmap(func, tqdm(inputs, total = len(inputs)))      

        self._ending_job()

    def _ending_job(self):
        self.pool.close()
        self.pool.join()

    def get_result(self):
        return self.value_collection        

def keywords_extraction(idx, ws, candidates, model, value_collection):
    
    # model = select_backend('./model_files/ckip_bert-base-chinese', 'flair')
    
    doc_embedding = model.doc_embed([' '.join(ws)])
    word_embeddings = model.word_embed(candidates, ws)
 
    keywords = select_similarity(doc_embedding  = doc_embedding, 
                                 word_embeddings = word_embeddings, 
                                 candidates = candidates, 
                                 top_n = 10,
                                 method = 'basic',
                                 excluding_same_word = True)
    
    value_collection.append((idx, keywords))


if __name__ == '__main__':
    import re
    import time 
    from utils import read_dictionary 
    from backend import select_backend
    from similarity_algorithm import select_similarity
    from preprocess import Ckip_Transformers_Tokenizer, Keyword_Candidates

    doc = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範s例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """
    docs = [doc]* 10
    tokenizer_model ='./model_files/ckip_bert-base-chinese_ws/'
    embedding_model = './model_files/ckip_bert-base-chinese'
    stop_words = read_dictionary('./dict/stop_words.txt')
    tokenizer = Ckip_Transformers_Tokenizer(tokenizer_model,use_device = 'cuda')

    ws_list = []
    candidates_list = []
    for d in tqdm(docs):
        # d = docs[0]
        ws, pos = tokenizer.tokenize(d)
        candidates = Keyword_Candidates(
                        stop_words = stop_words,
                        keyphrase_ngram_range = (1,7),
                        segment_by_stop_words = True,
                        excluding_stop_words = True,
                        min_df = 2,
                        pos_pattern = None,
                        including_pos_pattern = False
            ).extract_candidates(ws, pos)
        
        ws_list.append(ws)
        candidates_list.append(candidates)
    
    default_args = {}
    inputs = list(zip(range(len(ws_list)),ws_list,candidates_list))
    model = select_backend('./model_files/ckip_bert-base-chinese', 'flair')
    default_args['model'] = model

    process = Parallel_Computing(n_process = 6)
    process.run(keywords_extraction, inputs, default_args = default_args)
    print(process.get_result())
