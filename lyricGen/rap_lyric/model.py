import torch
import torch.nn as nn


class RapLyricGen(nn.Module):
    
    def __init__(self, num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size):
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = lr
        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, num_hidden, num_layers, dropout = drop_prob, batch_first = True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(num_hidden, vocab_size)      
    
   
    def forward(self, x, hidden):

        embedded = self.embedded(x)     
        lstm_output, hidden = self.lstm(embedded, hidden)
        dropout_out = self.dropout(lstm_output).reshape(-1, self.num_hidden) 
        out = self.fc(dropout_out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.num_hidden).zero_(),
                  weight.new(self.num_layers, batch_size, self.num_hidden).zero_())
        return hidden