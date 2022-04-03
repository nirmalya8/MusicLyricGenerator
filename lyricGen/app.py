from tabnanny import check
import gradio as gr 
import torch 
from model import LyricGenModel
import argparse
from dataset import Dataset
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--sequence-length', type=int, default=10)
args = parser.parse_args()
dataset = Dataset(args)
model = LyricGenModel(dataset)
model.load_state_dict(torch.load("lyric_gen2.pt"))
def predict(textinp,check, next_words=100): #dataset, model,
    model.eval()
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
        state_h, state_c = model.init_state(len(words))
        for i in range(0, next_words):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
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
textbox = gr.inputs.Textbox(lines=1, placeholder="Let's generate lyrics", default="", label="Enter a word")#, optional=True)
checkbox = gr.inputs.CheckboxGroup(listofwords,label='Or, choose either of these',type="value")#, optional=False)


# print(checkbox)
gui = gr.Interface(fn=predict, #callable function
                   inputs=[textbox,checkbox], # or checkbox, #input format
                   outputs="text")

gui.launch(debug=True)
