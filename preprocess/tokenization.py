import torch
from utils import read_dictionary 
from abc import ABC, abstractmethod
from ckip_transformers.nlp import CkipWordSegmenter

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, texts):
        raise NotImplementedError

    def excluding_stop_words(self, texts):
        stop_words = read_dictionary('./dict/stop_words.txt')

        text_without_stop_words = []
        for idx, text in enumerate(texts):
            text_without_stop_words.append([ s for s in text if s.strip() not in stop_words])

        return text_without_stop_words     

class Ckip_Transformers_Tokenizer(Tokenizer):
    def __init__(self, model_path, use_device):
        
        if use_device == 'cuda':
            if torch.cuda.is_available():
                self.device = 0
            else :
                self.device = -1
        else:
            self.device = -1
        
        self.ws_model = CkipWordSegmenter(model_name = model_path,
                                          level = 3, device = self.device)

    def tokenize(self, texts, stop_words = False):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            pass 
        else:
            raise ValueError(
                f"Expect text type (str or List[str]) but got {type(texts)}"
            )   
        ws_result = self.ws_model(texts)
        
        if stop_words:
            ws_result = self.excluding_stop_words(ws_result)
        
        return ws_result

if __name__ == '__main__':
    text = """
        監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
        [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
        [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
        監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
        最佳方案將使演算法能夠正確確定未見實例的類標籤。
        這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
    """
    WS = Ckip_Transformers_Tokenizer('./model_files/ckip_bert-base-chinese_ws/', use_device = 'cuda')
    result = WS.tokenize(text, stop_words=True)
