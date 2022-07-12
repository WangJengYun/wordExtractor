import re
from tqdm import  tqdm
from utils import read_dictionary 
from backend import select_backend
from similarity_algorithm import select_similarity
from preprocess import Ckip_Transformers_Tokenizer, Keyword_Candidates
from multiprocessing import Manager, Pool

def keywords_extraction(idx, ws, candidates, model, M):
    
    # model = select_backend('./model_files/ckip_bert-base-chinese', 'flair')
    
    doc_embedding = model.doc_embed([' '.join(ws)])
    word_embeddings = model.word_embed(candidates, ws)
 
    keywords = select_similarity(doc_embedding  = doc_embedding, 
                                 word_embeddings = word_embeddings, 
                                 candidates = candidates, 
                                 top_n = 10,
                                 method = 'basic',
                                 show_info = False)
    
    M.append((idx, keywords))

def test(idx,M):
    M['idx' + str(idx)] = idx

if __name__ == '__main__':

    doc = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範s例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """
    docs = [doc]* 1000
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
                        pos_pattern = None
            ).extract_candidates(ws, pos)
        
        ws_list.append(ws)
        candidates_list.append(candidates)
    
    model = select_backend('./model_files/ckip_bert-base-chinese', 'flair')
    
    for idx, (ws, candidates) in enumerate(tqdm(list(zip(ws_list,candidates_list)))):
        A = []
        keywords_extraction(idx, ws, candidates, model, A)
    
    pool = Pool(5)   
    M = Manager().list() 
    pbar = tqdm(total = len(ws_list))
    update = lambda *args:pbar.update()
    for idx, (ws, candidates) in enumerate(list(zip(ws_list,candidates_list))):
        pool.apply_async(keywords_extraction, args = (idx, ws, candidates, model, M), callback=update)
    
    pool.close()
    pool.join()
    
    print(M)

            
        