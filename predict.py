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
import data_loader_0703 as dl
import Transformer as TF
import matplotlib.pyplot as plt


def predict(model,data,device):
    model.eval()
    j_word_count=14
    text, dec_input, target = data["text"], data["dec_input"], data["dec_target"]
    text, dec_input, target = text.to(device), dec_input.to(device), target.to(device)
    mask = nn.Transformer.generate_square_subsequent_mask(j_word_count + 1).to(device)   #マスクの作成

    outputs = model(text, dec_input, mask)

    return outputs

def main():
    df = pd.read_excel("./data/JEC_basic_sentence_v1-3.xls", header = None)
    j_t = Tokenizer()
    e_t = spacy.load('en_core_web_sm')

    #各文章をトークンに変換
    texts = df.iloc[:,1].apply(lambda x: dl.j_tokenizer(x, j_t))
    targets = df.iloc[:,2].apply(lambda x: dl.e_tokenizer(x, e_t))
  

    j_v,enc_vocab_size = dl.create_jvocab(texts)
    e_v,dec_vocab_size = dl.create_evocab(targets)

    j_text_transform , e_text_transform = dl.numericalization(j_v,e_v)

    dataset = dl.Dataset(df, j_text_transform, e_text_transform,j_v,e_v,j_t,e_t)

    data_loader = dl.make_dataloader(dataset=dataset,BATCH_SIZE=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device} device\n")
    model = torch.load("./weight/transformer_test.pth")
    
    data = next(iter(data_loader)) #最初のバッチを取得するためのコード 

    outputs = predict(model,data,device)
    
    #print(outputs.shape,"\n") #torch.Size([8, 15, 6072])
    print("英語辞書の長さ",len(e_v),"日本語辞書の長さ",len(j_v),sep="\n")





    #入力した日本語
    print("\n日本語の入力\n")
    text = data["text"]
    batch_size, sequence_length = text.shape
    final_output = []
    for i in range(batch_size):
        sentence = [j_v.get_itos()[idx.item()] for idx in text[i]]
        final_output.append(sentence)

    # 結果を表示
    for i, sentence in enumerate(final_output):
        print(f"入力日本語　 {i+1}: {' '.join(sentence)}")


    #--------------英訳後-----------------
    print("\n英訳語の出力\n")

    # DecoderクラスでSoftmax層を挿入していないため、softmax関数を適用して確率に変換,最も確率の大きいトークンを出力
    probabilities = F.softmax(outputs, dim=-1)
    predicted_indices = torch.argmax(probabilities, dim=-1)

    # 変換後のテンソルの形状を確認
    #print("Shape of the probabilities tensor:", predicted_indices.shape)
    #print(predicted_indices)
    #print(predicted_indices.shape)

    batch_size, sequence_length = predicted_indices.shape
    final_output = []

    for i in range(batch_size):

        sentence = [e_v.get_itos()[idx.item()] for idx in predicted_indices[i]]
        final_output.append(sentence)

    # 結果を表示
    for i, sentence in enumerate(final_output):
        print(f"出力英語　 {i+1}: {' '.join(sentence)}")

if __name__ == "__main__":
    main()

