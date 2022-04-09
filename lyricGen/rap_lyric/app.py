import gradio as gr
import torch
import torch.nn.functional as Functional
import numpy as np
from preprocessing import get_preprocessed_data
from model import RapLyricGen
import random

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

def generate(start_text, num_words=60):
    
    # baseline model eval
    model.eval()
    
    # create the initial hidden layer of batch size 1
    hidden = model.init_hidden(1)
    
    # convert the starting text into tokens
    tokens = start_text.split()
    
    # iterate through and predict the next token
    try:
        for token in start_text.split():
            curr_token, hidden = predict(model, token, hidden)
    except KeyError:
        print(','.join(uniq_words))
        start = random.randint(0, len(uniq_words))
        stop = len(uniq_words) if start+10 >= len(uniq_words) else start + 10
        return f"Word not in vocabulary, try one of: {','.join(uniq_words[start:stop])} "
    
    # add the token
    tokens.append(curr_token)
    
    # predict the subsequent tokens
    for token_num in range(num_words - 1):
        token, hidden = predict(model, tokens[-1], hidden)
        tokens.append(token)
        
    # return the formatted string
    return " ".join(tokens)

_,_,vocab_size,word_to_idx,idx_to_word, uniq_words= get_preprocessed_data()
num_hidden = 256
num_layers = 4
embed_size = 200
drop_prob = 0.3
lr = 0.001
num_epochs = 50
batch_size = 32
model = RapLyricGen(num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size)
model.load_state_dict(torch.load("rap_gen_3.pt"))

#print(generate(model,"chicago"))

start = random.randint(0, len(uniq_words))
stop = len(uniq_words) if start+5 >= len(uniq_words) else start + 5
random.shuffle(uniq_words)
#unique_words = ", ".join(unique_words)
#print(unique_words)
textbox = gr.inputs.Textbox(lines=1, placeholder=f"Let's generate lyrics, enter a word or try one of the words below", default="", label="Enter a word")#, optional=True)
#checkbox = gr.inputs.CheckboxGroup(listofwords,label='Or, choose either of these',type="value")#, optional=False)


# print(checkbox)
gui = gr.Interface(fn=generate, #callable function
                   inputs=textbox, # or checkbox, #input format
                   outputs="text",theme="dark-grass",
                   title="Rap Lyrics Generator",
                   description="This lyrics generator is based on Kanye West's songs. P.S. You might get some slangs here and there. Enjoy!",
                   examples = list(uniq_words[start:stop]) #[["hate you"],["add"],["rhymes"]]
                   )

gui.launch()