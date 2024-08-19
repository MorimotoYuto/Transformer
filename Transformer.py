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
import data_loader_0703 as dl
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)).to(device)
    pe = torch.zeros(max_len, 1, dim).to(device)
    pe[:, 0, 0::2] = torch.sin(position * div_term) #idx=0から２つおきに取り出す　ベクトル次元が偶数ならsin
    pe[:, 0, 1::2] = torch.cos(position * div_term) #idx=1から２つおきに取り出す　ベクトル次元が奇数ならcos
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)     #汎用化

class MultiHeadAttention(nn.Module):
  
  def __init__(self, dim, head_num, dropout = 0.1):  #dim はモデルの特徴量の次元数 例えば、単語やトークンの埋め込みベクトルの次元数   head_nuはマルチヘッドの数
    super().__init__() 
    self.dim = dim
    self.head_num = head_num
    self.linear_Q = nn.Linear(dim, dim, bias = False)
    self.linear_K = nn.Linear(dim, dim, bias = False)
    self.linear_V = nn.Linear(dim, dim, bias = False)
    self.linear = nn.Linear(dim, dim, bias = False)
    self.soft = nn.Softmax(dim = 3) #テンソルの三番目の次元(トークン)についてsoftmax
    self.dropout = nn.Dropout(dropout)
  
  def split_head(self, x):#マルチヘッド数に分ける
    x = torch.tensor_split(x, self.head_num, dim = 2) #埋め込みベクトル(512)は第二次元
    x = torch.stack(x, dim = 1)
    return x
  
  def concat_head(self, x):
    x = torch.tensor_split(x, x.size()[1], dim = 1)
    x = torch.concat(x, dim = 3).squeeze(dim = 1)
    return x

  def forward(self, Q, K, V, mask = None):
    Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)  W_Q     Q : n（シークエンス数）* d_model(512)の行列
    K = self.linear_K(K)   # W_K
    V = self.linear_V(V)
    
    Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count//head_num,dim)  Q : n * d_q(64)の行列　 (512//8=64)
    K = self.split_head(K)
    V = self.split_head(V)


    #アテンション機構
    QK = torch.matmul(Q, torch.transpose(K, 3, 2))    
    QK = QK/((self.dim//self.head_num)**0.5)

    if mask is not None:
      QK = QK + mask

    softmax_QK = self.soft(QK)
    softmax_QK = self.dropout(softmax_QK)

    QKV = torch.matmul(softmax_QK, V)
    QKV = self.concat_head(QKV)
    QKV = self.linear(QKV)
    return QKV

class FeedForward(nn.Module):

  def __init__(self, dim, hidden_dim = 2048, dropout = 0.1):
    super().__init__() 
    self.dropout = nn.Dropout(dropout)
    self.linear_1 = nn.Linear(dim, hidden_dim)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(hidden_dim, dim)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)
    return x

class EncoderBlock(nn.Module):

  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim])
    self.layer_norm_2 = nn.LayerNorm([dim])
    self.FF = FeedForward(dim)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x):
    Q = K = V = x
    x = self.MHA(Q, K, V)
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_2(x)
    x = x + _x
    x = self.layer_norm_2(x)
    return x

class Encoder(nn.Module):

  def __init__(self, enc_vocab_size, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.dim = dim
    self.embed = nn.Embedding(enc_vocab_size, dim)
    self.PE = PositionalEncoding(dim)
    self.dropout = nn.Dropout(dropout)
    self.EncoderBlocks = nn.ModuleList([EncoderBlock(dim, head_num) for _ in range(6)])

  def forward(self, x):
    x = self.embed(x)
    x = x*(self.dim**0.5)     #埋め込みベクトルが小さくなりすぎるのを防ぎ、勾配消失を防ぐ
    x = self.PE(x)
    x = self.dropout(x)
    for i in range(6):
      x = self.EncoderBlocks[i](x)
    return x

class DecoderBlock(nn.Module):

  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.MMHA = MultiHeadAttention(dim, head_num)
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim])
    self.layer_norm_2 = nn.LayerNorm([dim])
    self.layer_norm_3 = nn.LayerNorm([dim])
    self.FF = FeedForward(dim)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)

  def forward(self, x, y, mask):
    Q = K = V = x
    x = self.MMHA(Q, K, V, mask)
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    Q = x
    K = V = y
    x = self.MHA(Q, K, V)
    x = self.dropout_2(x)
    x = x + Q
    x = self.layer_norm_2(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_3(x)
    x = x + _x
    x = self.layer_norm_3(x)
    return x

class Decoder(nn.Module):

  def __init__(self, dec_vocab_size, dim, head_num, dropout = 0.1):
    super().__init__() 
    self.dim = dim
    self.embed = nn.Embedding(dec_vocab_size, dim)
    self.PE = PositionalEncoding(dim)
    self.DecoderBlocks = nn.ModuleList([DecoderBlock(dim, head_num) for _ in range(6)])
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(dim, dec_vocab_size)

  def forward(self, x, y, mask):
    x = self.embed(x)
    x = x*(self.dim**0.5)
    x = self.PE(x)
    x = self.dropout(x)
    for i in range(6):
      x = self.DecoderBlocks[i](x, y, mask)
    x = self.linear(x)   #損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない
    return x

class Transformer(nn.Module):
  
  def __init__(self, enc_vocab_size, dec_vocab_size, dim, head_num):
    super().__init__() 
    self.encoder = Encoder(enc_vocab_size, dim, head_num)
    self.decoder = Decoder(dec_vocab_size, dim, head_num)

  def forward(self, enc_input, dec_input, mask):
    enc_output = self.encoder(enc_input)
    output = self.decoder(dec_input, enc_output, mask)
    return output



def main():
  # 辞書のサイズ（例えば、10000語）
  vocab_size = 10000
  # 各埋め込みベクトルの次元数
  embedding_dim = 300

  # nn.Embeddingのインスタンス化
  embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

  # 任意のインデックスのテンソル（バッチサイズ2、シーケンス長3）
  input_indices = torch.tensor([[1, 2, 3], [4, 5, 6],[1, 9, 6]])

  # 埋め込み層を通してインデックスを埋め込みベクトルに変換
  embedded_vectors = embedding_layer(input_indices)

  print(embedded_vectors)



  enc_vocab_size = 4000
  dec_vocab_size = 3900
  transformer = Transformer(enc_vocab_size, dec_vocab_size, dim = 512, head_num = 8).to(device)
  for name, param in transformer.named_parameters():
    if name == "decoder.embed.weight":  # デコーダーの埋め込み層のパラメーター
      print(f"name:{name}  param:{param}\n")


if __name__ == "__main__":
    main()
