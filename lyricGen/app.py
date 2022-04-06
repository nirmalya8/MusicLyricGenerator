from tabnanny import check
import gradio as gr 
import torch 
from model import LyricGenModel
import argparse
from dataset import Dataset
import numpy as np
import random
import kanye_rap.preprocessing.get_preprocessed_data as get_preprocessed_data
import kanye_rap.model.RapLyricGen as RapLyricGen
import os
import torch.nn.functional as Functional

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--sequence-length', type=int, default=10)
args = parser.parse_args()
dataset = Dataset(args)
model1 = LyricGenModel(dataset)
model1.load_state_dict(torch.load("models\lyric_gen2.pt"))

_,_,vocab_size,word_to_idx,idx_to_word = get_preprocessed_data()
num_hidden = 256
num_layers = 4
embed_size = 200
drop_prob = 0.3
lr = 0.001
num_epochs = 50
batch_size = 32
model2 = RapLyricGen(num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size)
model2.load_state_dict(torch.load(os.path.join("models","rap_gen_3.pt")))


def predict(textinp,check, next_words=100): #dataset, model,
    model1.eval()
    try:
        print(type(textinp))
        print(textinp)
        print(type(check))
        print(check)
        text = None
        if textinp == '':
             text = check
        elif check == []:
             text = textinp
        else:
             text = textinp
        print(text)
        words = ""
        if type(text) == list:
            #print(words)
            words = str(text[0])
            print(words)
            #words = words[0]
        else:
            words = text.split(' ')
        print(words)
        state_h, state_c = model1.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
            y_pred, (state_h, state_c) = model1(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(dataset.index_to_word[word_index])
        return " ".join(words)
    except KeyError:
        unique_words = dataset.get_uniq_words()
        print(','.join(unique_words))
        start = random.randint(0, len(unique_words))
        stop = len(unique_words) if start+10 >= len(unique_words) else start + 10
        return f"Word not in vocabulary, try one of: {','.join(unique_words[start:stop])} "

unique_words = dataset.get_uniq_words()
start = random.randint(0, len(unique_words)-4)
stop = start + 3
listofwords = unique_words[start:stop]
for word in listofwords:
    if word==',':
        listofwords.remove(word)
print(listofwords)


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

textbox = gr.inputs.Textbox(lines=1, placeholder="Let's generate lyrics", default="", label="Enter a word")#, optional=True)
checkbox = gr.inputs.CheckboxGroup(['Rap Music','Country Music'],label='Choose one of these genres',type="value")#, optional=False)



# print(checkbox)
gui = gr.Interface(fn=predict, #callable function
                   inputs=[textbox,checkbox], # or checkbox, #input format
                   outputs="text")

gui.launch(debug=True)
