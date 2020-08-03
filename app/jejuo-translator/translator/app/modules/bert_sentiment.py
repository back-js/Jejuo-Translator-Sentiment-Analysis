import torch
import random
import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']

device = torch.device('cpu')

from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-multilingual-cased')

import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim,
                n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                         num_layers = n_layers,
                         bidirectional = bidirectional,
                         batch_first = True,
                         dropout = 0 if n_layers <2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional
                            else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]    

        _, hidden = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            # 마지막 레이어의 양방향 히든 벡터만 가져옴
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        output = self.out(hidden)
        
        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
model = model.to(device)

model.load_state_dict(torch.load('/content/drive/My Drive/jejuo-translator/translator/app/model/sentiment_BERT_data/tut6-model.pt',map_location=device))

def predict_sentiment(sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    x = prediction.item()

    if x > 0.8 :
        x1 = '긍정입니다.'
        return x1
    elif x < 0.1 :
        x2 = '부정입니다.'
        return x2
    else :
        x3 = '중립입니다.'
        return x3

    
