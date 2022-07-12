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


doc = """
    俄烏戰爭長期化 全球需為外溢效應做準備
    中央社記者田習如台北28日電）雖然G7峰會昨天傳出烏克蘭總統澤倫斯基敦促大國領袖們「在年底前」讓俄羅斯停止侵略，但多數專家都已提出要為俄烏戰爭的長期化做好準備，包括因應國際戰略格局和經濟資源等的外溢效應。
澤倫斯基（Volodymyr Zelenskyy）透過視訊向七大工業國集團（G7）領袖說，他尚未準備好與俄國展開協商。烏克蘭的立場加上俄軍在G7峰會期間轟炸基輔和烏克蘭中部的平民設施，在在顯示短期內看不到和談契機；而除非蒲亭（Vladimir Putin）政權意外垮台，也看不到俄國主動喊停的可能。
歐盟部長理事會的安全顧問、未來學家高布（Florence Gaub）日前在美國政治新聞網站Politico的Podcast訪問中指出，統計過去200年的戰爭史，國與國的戰爭平均持續15個月，當然也有例外如兩伊戰爭打了8年等。
俄烏戰爭展示了從民間網路動員到國際經濟封鎖等非傳統軍武的面向，華府智庫戰略暨國際研究中心（CSIS）榮譽主席柯迪斯曼（Anthony H. Cordesman）認為：「想要讓衝突能夠以一種帶來持續和平的方式終結，正變得日益困難。」
他的意思是即便停火，經濟和民間的衝擊仍會持續，因為俄國和歐洲的政治緊張不止，更別說俄烏人民間的仇恨將延續至少一個世代。
柯迪斯曼在CSIS撰文指出幾個俄烏戰爭帶來的「外溢效應」：一、在俄國和北約、歐盟、美國間的緊張關係下，軍備競賽將達5年或更久，使軍費占GDP（國內生產毛額）的比重拉高好幾個百分點，對經濟帶來衝擊；二、中國將全力發展從政治到經濟上可以抵抗美國封鎖的能力；三、北韓、巴基斯坦、伊朗、土耳其以及其他在區域內有軍事力量的第三世界小國，都將捲入東西兩陣營的對抗中，包括貿易投資、軍事基地和防衛協助等。
經濟合作暨發展組織（OECD）6月的一份報告也分析這場戰爭帶來的中長期影響，包括改變全球能源市場結構、外匯存底配置結構，以及供應鏈重組等。報告指出，世界重新分割成互設壁壘的陣營後，專業（生產）分工、規模經濟、資訊和技術的分享、單一主導貨幣…這些過去帶來經濟效率的好處都將縮減。
OECD已在6月將全球今年經濟成長的預估值從原本的4.5%下修到3%，明年更預估只有2.8%。
因此，俄烏戰爭的拖長，包括持續一段長期的打打停停，或者即便達成停火協議但緊繃態勢長期持續下，不只歐美，全球許多國家都得因應資源的重新配置，例如國防預算上升的排擠效應、製造業供應鏈乃至市場布局的劃清界線、外交上拉幫結派的重新洗牌等。
美國前外交官哈斯（Richard Haass）6月上旬在「外交事務」（Foreign Affairs）撰文說，（俄烏戰爭）不論勝利或妥協在可預見的未來都被排除之下，美國和歐洲當局需要一個能夠管理「開放式衝突」的策略。他強調：是「管理」而非「解決」，因為真要解決得由俄國從根本改變行為，這在目前難以期待。
管理一場可能拖長的危機，而非等待衝突被解決，對於全球受到這場戰爭外溢效應波及的政府乃至企業決策者而言，也是最務實的因應之道。（編輯：陳惠珍）1110628
"""
   
kw_model = KeyWordExtractor(embedding_model = './model_files/ckip_bert-base-chinese', 
                            tokenizer_model ='./model_files/ckip_bert-base-chinese_ws/',
                            pos_model = None,
                            backend = 'flair',
                            stop_words = stop_words,
                            device = 'cuda')
   
keywords = kw_model.extract_keywords(doc, 
                                    keyphrase_ngram_range = (1, 10),
                                    segment_by_stop_words = True,
                                    min_df = 2,
                                    similarity_method = 'basic',
                                    top_n = 10,
                                    pos_pattern = None,
                                    excluding_same_word = False)


doc = """
    昇達科攜雷捷 大啖5G商機
5G通訊毫米波小基站為時勢所趨，看準此商機，昇達科與其轉投資之新創毫米波IC設計公司雷捷電子攜手開發5G通訊毫米波小基站高效能前端模組已有所成，昇達科指出，相關開發作業已依原定時程、於今年7月初完成產品開發，並經經濟部技術處評審委員一致通過順利結案。
5G時代來臨，根據最新的市場研究報告指出，全球5G小基站市場規模將在2028年達到 179億美元，5G基礎建設的部署正在加速，全球移動通訊系統（GSMA）統計，2019年 5G 連線數僅1000 萬，但到2025年將快速成長至20億左右，5G小基站的市場增長不容小覷。
看好此商機，昇達科攜手持股14.5％的子弟兵雷捷電子參與前瞻技術研發計畫，由昇達科負責毫米波39 GHz波導陣列天線與模組開發測試，而雷捷電子則負責毫米波收發端IC設計。昇達科指出，此前瞻技術研發計畫係於108年第四季獲得經濟部技術處審查通過並補助，計畫總經費為新台幣1.7億元。
昇達科指出，此計畫以鎖定5G毫米波小基站IAB (Integrated Access & Backhaul) 之前端陣列天線模組進行研究開發，此前端模組擁有體積小、低耗能、高散熱、低雜訊指數之性能，能提供國內外通信系統廠商兼顧傳輸效率與成本優勢的毫米波小基站IAB 前端模組解決方案。過去這類毫米波IC主要仰賴國外知名IC設計公司，而此次結合國內新創毫米波IC設計公司雷捷電子在毫米波 IC 研發設計的能力，開發出國產毫米波晶片與前端模組，未來將可藉由昇達科，與國際電信設備大廠在毫米波回傳元件與天線等產品長期合作、協同開發經驗，預期可望協助雷捷電子切入5G 毫米波基站之供應鏈，提供全球通信系統廠更具競爭力的優質選擇。
"""
   
kw_model = KeyWordExtractor(embedding_model = './model_files/ckip_bert-base-chinese', 
                            tokenizer_model ='./model_files/ckip_bert-base-chinese_ws/',
                            pos_model ='./model_files/ckip_bert-base-chinese_pos/',
                            backend = 'flair',
                            stop_words = stop_words,
                            device = 'cuda')
   
keywords = kw_model.extract_keywords(doc, 
                                    keyphrase_ngram_range = (1, 10),
                                    segment_by_stop_words = True,
                                    min_df = 4,
                                    similarity_method = 'basic',
                                    top_n = 20,
                                    pos_pattern = [(1, ['VC','FW'])],
                                    excluding_same_word = True)

doc = """
    台積電股價跌跌不休　楠梓設廠卻帶動房價連漲11個月
外資今年大賣台股，也讓有護國神山美名的台積電股價重挫，從今年1月17日曾站上688元，至7月5日出現盤中最低破底僅剩433元，僅6個月時間跌幅超過37%，然而相對股價跌跌不休，從去年6月起高市楠梓區受到台積電將「高雄煉油廠」舊址設廠，導致區段房價出現連漲11個月，漲幅達41%。專家分析，楠梓房價具備支撐主因除建設利多亦有人口紅利，相對股價暴跌房價具支撐。
楠梓區過去數十年來新屋房價多屬1字頭，該現象在2021年上半年都還是如此，許多大樓預售案成交單價落在16~19萬元，然而該現象在去年6月起開始出現變化，新屋一下子單坪大漲6~8萬元，導致房價全面站穩2字頭行情，在去年上半年購屋的民眾均享有房價大幅增值。《蘋果新聞網》委託台灣房屋集團趨勢中心調查，從去年6月起至今年4月，實登揭露楠梓區房價已連漲11個月，屋齡5年內建物，每坪均價從20.4萬元提升到28.8萬元，房價增幅達41%。
台灣房屋集團趨勢中心經理李家妮分析，這波台積電設廠效應，讓楠梓區甚至整個高雄出現過去幾年南科效應，預售屋每坪大漲6~8萬元，部分甚至出現10萬元上揚行情，帶動周邊中古屋房價，以房市交易熱區高雄大學生活圈為例，目前屋齡10年內大樓，房價已全面站上2字頭成交價，而新屋更有挑戰3字頭市況，即便近期房市買氣不及去年，但價格並未出現下修。 
楠梓區新屋房價不僅挑戰3字頭，甚至不少建案開價突破4字頭，如高雄大學特區大樓預售案「藍田玉」因屬面藍田公園首排景觀宅，每坪開價34~42萬元，位於後勁溪河岸首排大樓成屋案「觀雲3」，屬少見成屋且具備永久景觀棟距，每坪開價32~44萬元！
上宸國際總經理林毅表示，若以南科效應為借鏡，台積電所坐落的善化區，新屋房價如「桂田磐古2期」已出現成交4字頭，而這波台積電漲幅讓整個楠梓區新屋房價全面3字頭，就善化與楠梓生活機能相比，楠梓生活機能明顯較好，加上楠梓從縣市合併後17.31萬人增加至目前18.98萬元，具備人口紅利支撐。（葉家銘／高雄報導）
"""
   
kw_model = KeyWordExtractor(embedding_model = './model_files/ckip_bert-base-chinese', 
                            tokenizer_model ='./model_files/ckip_bert-base-chinese_ws/',
                            pos_model ='./model_files/ckip_bert-base-chinese_pos/',
                            backend = 'flair',
                            stop_words = stop_words,
                            device = 'cuda')
   
keywords = kw_model.extract_keywords(doc, 
                                    keyphrase_ngram_range = (1, 10),
                                    segment_by_stop_words = True,
                                    min_df = 2,
                                    similarity_method = 'basic',
                                    top_n = 20,
                                    pos_pattern = None,
                                    excluding_same_word = True)

















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
ws = ''

doc = Sentence(doc)
doc_embed_model.embed(doc)

doc = Sentence(doc)
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