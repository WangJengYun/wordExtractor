import torch
import numpy as np 
from utils import read_dictionary 
from abc import ABC, abstractmethod
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, texts):
        raise NotImplementedError

    def excluding_stop_words(self, texts, stop_words):

        text_index_without_stop_words = []
        for idx, text in enumerate(texts):
            text_index_without_stop_words.append([ idx for idx, s in enumerate(text) if s.strip() not in stop_words])

        return text_index_without_stop_words     

class Ckip_Transformers_Tokenizer(Tokenizer):
    def __init__(self, ws_model_path, pos_model_path = None, use_device = 'cuda'):
        
        if use_device == 'cuda':
            if torch.cuda.is_available():
                self.device = 0
            else :
                self.device = -1
        else:
            self.device = -1
        
        self.ws_model = CkipWordSegmenter(model_name = ws_model_path,
                                          level = 3, device = self.device)
        if pos_model_path is not None:
            self.pos_model = CkipPosTagger(model_name = pos_model_path,
                                          level = 3, device = self.device)
        else:
            self.pos_model = None 

    def tokenize(self, texts, stop_words = None):
        ws_result, pos_result = None, None 

        if isinstance(texts, str):
            texts = [texts]
            ws = self.ws_model(texts)[0]
            if stop_words is not None:
                ws_index = self.excluding_stop_words([ws], stop_words)[0]
                ws_result = np.array([ws])[0][ws_index].tolist()
            else:
                ws_result = ws 
            
            if self.pos_model is not None:
                pos = self.pos_model([ws])[0]
                if stop_words is not None:
                    pos_result = np.array([pos])[0][ws_index].tolist()
                else:
                    pos_result = pos 
  
        elif isinstance(texts, list):
            pass 
        else:
            raise ValueError(
                f"Expect text type (str or List[str]) but got {type(texts)}"
            )           
        return ws_result, pos_result

if __name__ == '__main__':
    text = """
        監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
        [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
        [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
        監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
        最佳方案將使演算法能夠正確確定未見實例的類標籤。
        這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """
    WS = Ckip_Transformers_Tokenizer(ws_model_path = './model_files/ckip_bert-base-chinese_ws/',
                                     pos_model_path = './model_files/ckip_bert-base-chinese_pos/', use_device = 'cpu')
    result = WS.tokenize(text, stop_words = None)
    # result = WS.tokenize(text, stop_words = read_dictionary('./dict/stop_words.txt'))

