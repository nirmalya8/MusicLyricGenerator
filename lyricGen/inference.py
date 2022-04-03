import torch 
import numpy as np
from model import LyricGenModel
import argparse
from dataset import Dataset

you_output = ['You', '\n', 'Can', 'we', 'always', 'be', 'this', 'close?', '\n', 'Forever', 'and', 'ever,', 'ah', '\n', 'Take', 'me', 'out,', 'and', 'take', 'me', 'home', '\n', "You're", 'my,', 'my,', 'my,', 'my', 'lover', '\n', 'Ooh,', 'look', 'what', 'you', 'made', 'me', 'do', '\n', 'Look', 'what', 'you', 'just', 'made', 'me', 'do', '\n', 'Look', 'what', 'you', 'made', 'me', 'do', '\n', 'Look', 'what', 'you', 'just', 'made', 'me', 'do', '\n', 'Look', 'what', 'you', 'made', 'me', 'tell', 'and', 'me', 'in', 'me', '\n', 'Is', 'me', 'do', '\n', 'Look', 'what', 'you', 'just', 'made', 'me', 'do', '\n', 'Ooh,', 'look', 'that', 'you', 'are', 'a', 'movie', '\n', 'Look', 'what', 'you', 'just', 'made', 'me', 'see', '\n', 'Look', 'what']
print(" ".join(you_output))
def predict(dataset, model, text, next_words=100):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words
parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--sequence-length', type=int, default=10)
args = parser.parse_args()
dataset = Dataset(args)
model = LyricGenModel(dataset)
model.load_state_dict(torch.load("lyric_gen2.pt"))
    
# w = predict(dataset, model, text='Let')
# print(" ".join(w))

w = predict(dataset, model, text='Why')
print(" ".join(w))
# print(" ".join(predict(dataset, model,text='Let Me')))
# print(" ".join(predict(dataset, model,text="Why")))

# Why 
#  Can we made why we ever, be this close?
#  ever, ah
#  Take of ah
#  Look what you made me do
#  Look what you made me do
#  Look what you just made me—
#  Ooh, look what you made me do
#  Look what you just made me do
#  Look what you just made me do
#  She's oh-oh, oh-oh, oh-oh, oh-oh
#  You need to calm down
#  In the middle of the night, in my dreams
#  I just see what you need home
#  I shoulda known
#  Losing him was

# Why always me 
#  She's girl up to walk away 
#  If you love me then I feel it way 
#  Since the beginning for the teardrops on my guitar 
#  The only thing that keeps me wishing on a wishing star 
#  He's the song in the car 
#  I keep singing, don't know a movie 
#  Unsure of turning you away 
#  Wondering you find somebody girl, girl 
#  But I'll be right here on the ground so oh) 
#  Don't say it will be a lonely time 
#  I let it wrote 
#  As it love around the seventeenth 