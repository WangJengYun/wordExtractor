from ast import Not
import re 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

class Keyword_Candidates(object):

    def __init__(self, 
        stop_words,
        keyphrase_ngram_range,
        segment_by_stop_words,
        excluding_stop_words,
        min_df,
        pos_pattern,
        including_pos_pattern,
        excluding_specific_words):
        
        self.stop_words = stop_words
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.segment_by_stop_words = segment_by_stop_words
        self.excluding_stop_words = excluding_stop_words
        self.min_df = min_df
        self.pos_pattern = pos_pattern
        self.including_pos_pattern = including_pos_pattern
        self.excluding_specific_words = excluding_specific_words

        if self.segment_by_stop_words or self.excluding_stop_words:
            assert stop_words is not None
        
        self.pos_pattern_dict = {}
        if self.pos_pattern is not None:            
            for n_gram, pos_tagger in self.pos_pattern:
                self.pos_pattern_dict['gram_' + str(n_gram)] = pos_tagger


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
                ngram_range = self.keyphrase_ngram_range, stop_words = input_stop_words, min_df = self.min_df,
                token_pattern = r"(?u)\b\w+\b"
                ).fit(sentences)

        candidates = count.get_feature_names()

        return candidates

    def _get_candidates_positions(self, ws, candidates):
        ws = [s.lower() for s in ws]
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
            if (len(positions)>= self.min_df) and len(phrase) > 1:
                words_positions.append((positions.tolist(), n_split))
                words.append(phrase)
        
        return words, words_positions
    
    def _filter_pos(self, pos, candidates_info):
        info = []
        pos = np.array(pos)
        
        for word, (positions, n_gram) in candidates_info:
            
            position_list = [[ i + j for j in range(n_gram)] for i in positions]
            
            word_pos = []
            for gram_position in position_list:
                word_pos.append(' '.join(pos[gram_position]))
            
            gram_name = 'gram_' + str(n_gram)
            word_pos = list(set(word_pos))
            
            if self.pos_pattern_dict and (gram_name in self.pos_pattern_dict.keys()):
                if len(word_pos) == 1:
                    if self.including_pos_pattern:
                        if word_pos[0] in self.pos_pattern_dict[gram_name]:
                            info.append((word, (positions, n_gram, word_pos[0])))
                    else:
                        if word_pos[0] not in self.pos_pattern_dict[gram_name]:
                            info.append((word, (positions, n_gram, word_pos[0])))
                else:
                    if self.including_pos_pattern:
                        if any([p in  self.pos_pattern_dict[gram_name] for p in word_pos]):
                            info.append((word, (positions, n_gram, word_pos)))
                    else:   
                        if any([p in  self.pos_pattern_dict[gram_name] for p in word_pos]):
                            info.append((word, (positions, n_gram, word_pos))) 
            else:
                if len(word_pos) == 1:
                    info.append((word, (positions, n_gram, word_pos[0])))
                else:
                    info.append((word, (positions, n_gram, word_pos)))

        return info

    def _excluding_words(self, candidates):
        
        filter_candidates = []
        for idx, candidate in enumerate(candidates):
            word = candidate.replace(' ','')
            
            if word not in self.excluding_specific_words:
                filter_candidates.append(candidate)

        return filter_candidates                

    def extract_candidates(self, ws, pos):

        sentences = self._segment_process(ws)

        candidates = self._get_candidates(sentences)

        if self.excluding_specific_words is not None :
            candidates = self._excluding_words(candidates)
        
        candidates, candidates_positions = self._get_candidates_positions(ws, candidates)

        candidates_info = list(zip(candidates, candidates_positions))

        if pos is not None:
            candidates_info = self._filter_pos(pos, candidates_info)

        return candidates_info

if __name__ == '__main__':
    import os 
    os.chdir('c:\\Users\\cloudy822\\Desktop\\wordExtractor')

    from utils import read_dictionary 
    from preprocess.tokenization import Ckip_Transformers_Tokenizer
    
    stop_words = read_dictionary('./dict/stop_words.txt')
    
    doc = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """

    tokenizer = Ckip_Transformers_Tokenizer(ws_model_path = './model_files/ckip_albert-tiny-chinese-ws/' ,
                                            pos_model_path = './model_files/ckip_bert-base-chinese_pos/', use_device = 'cuda')
    ws,pos = tokenizer.tokenize(doc)

    KC = Keyword_Candidates(stop_words = stop_words,
                            keyphrase_ngram_range = (1, 5),
                            segment_by_stop_words = True,
                            excluding_stop_words = True,
                            min_df = 2,
                            pos_pattern = [(2, ['Na Na'])])

    candidates = KC.extract_candidates(ws, pos)
    