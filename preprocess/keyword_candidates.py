import re 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

class Keyword_Candidates(object):

    def __init__(self, 
        stop_words,
        keyphrase_ngram_range,
        segment_by_stop_words,
        excluding_stop_words,
        min_df):
        
        self.stop_words = stop_words
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.segment_by_stop_words = segment_by_stop_words
        self.excluding_stop_words = excluding_stop_words
        self.min_df = min_df

        if self.segment_by_stop_words or self.excluding_stop_words:
            assert stop_words is not None

    def _segment_process(self, ws):
        
        ws = [s.strip() for s in ws if s.strip() != '']
        
        sentences = None
        if self.segment_by_stop_words:
            segment_stop_words_pattern = ' | '.join([ re.escape(s) for s  in self.stop_words])  
            sentences = re.split(segment_stop_words_pattern, ' '.join(ws))      
        
        else:        
            sentences = ' '.join(ws)
        
        return sentences

    def _get_candidates(self, sentences):
        if self.excluding_stop_words:
            input_stop_words = self.stop_words
        else:
            input_stop_words = None 

        count = CountVectorizer(
                ngram_range = self.keyphrase_ngram_range, stop_words = input_stop_words, min_df = self.min_df
                ).fit(sentences)

        candidates = count.get_feature_names()

        return candidates

    def _get_candidates_positions(self, ws, candidates):
        ws = np.array(ws)
        words = []
        words_positions = []
        for phrase in candidates:    
            word_list = phrase.split(' ')
            n_split = len(word_list)
            n = 0
            positions = None  
            while n < n_split:
                if n == 0:
                    positions = np.where(ws == word_list[n])[0]
                else:
                    positions = positions[ws[positions + n] == word_list[n]]

                n += 1 
            if len(positions)>= self.min_df:
                words_positions.append((positions.tolist(), n_split))
                words.append(phrase)
        
        return words, words_positions

    def extract_candidates(self, ws):

        sentences = self._segment_process(ws)

        candidates = self._get_candidates(sentences)
        
        candidates, candidates_positions = self._get_candidates_positions(ws, candidates)

        return list(zip(candidates, candidates_positions))

if __name__ == '__main__':
    import os 
    os.chdir('c:\\Users\\cloudy822\\Desktop\\wordExtractor')

    from utils import read_dictionary 
    from tokenization import Ckip_Transformers_Tokenizer
    
    stop_words = read_dictionary('./dict/stop_words.txt')
    
    doc = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """

    tokenizer = Ckip_Transformers_Tokenizer('./model_files/ckip_albert-tiny-chinese-ws/' ,use_device = 'cuda')
    ws = tokenizer.tokenize(doc)[0]

    KC = Keyword_Candidates(stop_words = stop_words,
                            keyphrase_ngram_range = (1, 5),
                            segment_by_stop_words = True,
                            excluding_stop_words = True,
                            min_df = 2)

    candidates = KC.extract_candidates(ws)
    