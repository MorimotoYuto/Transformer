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
import pickle

def train(dataset,data_loader,enc_vocab_size, dec_vocab_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TF.Transformer(enc_vocab_size, dec_vocab_size, dim = 512, head_num = 8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 200
    print_coef = 10
    train_length = len(dataset)

    history = {"train_loss": []}
    n = 0
    train_loss = 0
    j_word_count = 14

    for i, data in enumerate(data_loader):


      for epoch in range(epoch_num):
        model.train()
        for i, data in enumerate(data_loader):
          optimizer.zero_grad()
          text, dec_input, target = data["text"].to(device), data["dec_input"].to(device), data["dec_target"].to(device)
          mask = nn.Transformer.generate_square_subsequent_mask(j_word_count + 1).to(device)   #マスクの作成

          outputs = model(text, dec_input, mask)
          target = nn.functional.one_hot(target, dec_vocab_size).to(torch.float32)

          loss = criterion(outputs, target)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          history["train_loss"].append(loss.item())
          n += 1
          if i % ((train_length//data_loader.BATCH_SIZE)//print_coef) == (train_length//data_loader.BATCH_SIZE)//print_coef - 1:
            print(f"epoch:{epoch+1}  index:{i+1}  loss:{train_loss/n:.10f}")
            train_loss = 0
            n = 0

    torch.save(model,"transformer_test_2.pth")

    with open("history.pkl","wb") as f:
        pickle.dump(history, f)
        
    plt.plot(history["train_loss"])
    plt.show()



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
    
    train()
 

if __name__ == "__main__":
    main()