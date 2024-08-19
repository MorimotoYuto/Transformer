import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
import torchtext.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import math
import janome
from janome.tokenizer import Tokenizer
import spacy
from collections import Counter



df = pd.read_excel("./data/JEC_basic_sentence_v1-3.xls", header = None)


#print(df.head())

#日本語用のトークン変換関数を作成
j_t = Tokenizer()
def j_tokenizer(text): 
    return [tok for tok in j_t.tokenize(text, wakati=True)]

#英語用のトークン変換関数を作成
e_t = spacy.load('en_core_web_sm')
def e_tokenizer(text):
    return [tok.text for tok in e_t.tokenizer(text)]

e_text = "I have a pen."
j_text = "私はペンを持っている"
#print(e_tokenizer(e_text),"\n",j_tokenizer(j_text))

#各文章をトークンに変換
texts = df.iloc[:,1].apply(j_tokenizer)
targets = df.iloc[:,2].apply(e_tokenizer)

print(texts,"\n",targets)


#-------------------------------------------　ここまでトークン化　---------------------------------------------------

#日本語のトークン数（単語数）をカウント
j_list = []
for i in range(len(texts)):
  j_list.extend(texts[i])
j_counter = Counter()
j_counter.update(j_list)
j_v = vocab(j_counter, specials=(['<unk>', '<pad>', '<bos>', '<eos>']))   #特殊文字の定義
j_v.set_default_index(j_v['<unk>'])

#英語のトークン数（単語数）をカウント
e_list = []
for i in range(len(targets)):
  e_list.extend(targets[i])
e_counter = Counter()
e_counter.update(e_list)
e_v = vocab(e_counter, specials=(['<unk>', '<pad>', '<bos>', '<eos>']))   #特殊文字の定義
e_v.set_default_index(e_v['<unk>'])

enc_vocab_size, dec_vocab_size = len(j_v), len(e_v)



#-------------------------------------　ここまで辞書化　-------------------------------------------





#各言語ごとに単語数を合わせる必要がある為、1文当たりの単語数を14に指定       数値化
j_word_count = 14
e_word_count = 14

j_text_transform = T.Sequential(
  T.VocabTransform(j_v),   #トークンに変換
  T.Truncate(j_word_count),   #14語以上の文章を14語で切る
  T.AddToken(token=j_v['<bos>'], begin=True),   #文頭に'<bos>
  T.AddToken(token=j_v['<eos>'], begin=False),   #文末に'<eos>'を追加
  T.ToTensor(),   #テンソルに変換
  T.PadTransform(j_word_count + 2, j_v['<pad>'])   #14語に満たない文章を'<pad>'で埋めて14語に合わせる
)

e_text_transform = T.Sequential(
  T.VocabTransform(e_v),   #トークンに変換
  T.Truncate(e_word_count),   #14語以上の文章を14語で切る
  T.AddToken(token=e_v['<bos>'], begin=True),   #文頭に'<bos>
  T.AddToken(token=e_v['<eos>'], begin=False),   #文末に'<eos>'を追加
  T.ToTensor(),   #テンソルに変換
  T.PadTransform(e_word_count + 2, e_v['<pad>'])   #14語に満たない文章を'<pad>'で埋めて14語に合わせる
)

#---------------------------------　ここまで数値化のためのトランス定義　----------------------------------



class Dataset(Dataset):    #トークン化と辞書を活用した数値化を行う
  def __init__(
      self,
      df,
      j_text_transform,
      e_text_transform,
      j_v,
      e_v,
      ):
    
    self.texts = df.iloc[:,1].apply(j_tokenizer)
    self.targets = df.iloc[:,2].apply(e_tokenizer)
    self.j_text_transform = j_text_transform
    self.e_text_transform = e_text_transform
    self.j_v = j_v
    self.e_v = e_v
  
  def max_word(self):
    return len(self.j_v), len(self.e_v)
        
  def __getitem__(self, i):
    text = self.texts[i]
    text = self.j_text_transform([text]).squeeze()

    target = self.targets[i]
    target = self.e_text_transform([target]).squeeze()

    dec_input = target[:-1]
    dec_target = target[1:]   #右に1つずらす
    data = {"text": text, "dec_input": dec_input, "dec_target": dec_target}
    return data
  
  def __len__(self):
    return len(self.texts)

dataset = Dataset(df, j_text_transform, e_text_transform,j_v,e_v)
print(dataset.max_word(),"\n",dataset[0],"\n",len(dataset)) #辞書サイズを返す、データセットの最初のtextを返す、データセットの長さ

#ーーーーーーーーーーーーーーーーーーーーーー　データセットの作成　ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
#ーーーーーーーーーーーーーーーーーーーーーーー　以下データローダー　ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
BATCH_SIZE = 8

data_loader = DataLoader(dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    drop_last=True,#データセットのサイズがバッチサイズで割り切れない場合、最後の不完全なバッチを無視
    shuffle=True)

data = next(iter(data_loader)) #最初のバッチを取得するためのコード
text, dec_input, target = data["text"], data["dec_input"], data["dec_target"]
print(text[0], dec_input[0], target[0], sep="\n")
print(data)
for i, data in enumerate(data_loader):
  if i == 0:
    print(i)
    print(data)
    text, dec_input, target = data["text"], data["dec_input"], data["dec_target"]
    print(text[0], dec_input[0], target[0], sep="\n")