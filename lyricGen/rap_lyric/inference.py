import torch
import torch.nn.functional as Functional
import numpy as np
from preprocessing import get_preprocessed_data
from model import RapLyricGen

def predict(model, tkn, hidden_layer):
         
    # create torch inputs
    x = np.array([[word_to_idx[tkn]]])
    inputs = torch.from_numpy(x).type(torch.LongTensor)

    # detach hidden state from history
    hidden = tuple([layer.data for layer in hidden_layer])

    # get the output of the model
    out, hidden = model(inputs, hidden)

    # get the token probabilities and reshape
    prob = Functional.softmax(out, dim=1).data.numpy()
    prob = prob.reshape(prob.shape[1],)

    # get indices of top 3 values
    top_tokens = prob.argsort()[-3:][::-1]
    
    # randomly select one of the three indices
    selected_index = top_tokens[0]

    # return word and the hidden state
    return idx_to_word[selected_index], hidden

def generate(model,start_text, num_words=60):
    
    # baseline model eval
    model.eval()
    
    # create the initial hidden layer of batch size 1
    hidden = model.init_hidden(1)
    
    # convert the starting text into tokens
    tokens = start_text.split()
    
    # iterate through and predict the next token
    for token in start_text.split():
        curr_token, hidden = predict(model, token, hidden)
    
    # add the token
    tokens.append(curr_token)
    
    # predict the subsequent tokens
    for token_num in range(num_words - 1):
        token, hidden = predict(model, tokens[-1], hidden)
        tokens.append(token)
        
    # return the formatted string
    return " ".join(tokens)

_,_,vocab_size,word_to_idx,idx_to_word = get_preprocessed_data()
num_hidden = 256
num_layers = 4
embed_size = 200
drop_prob = 0.3
lr = 0.001
num_epochs = 50
batch_size = 32
model = RapLyricGen(num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size)
model.load_state_dict(torch.load("rap_gen_3.pt"))

print(generate(model,"chicago"))