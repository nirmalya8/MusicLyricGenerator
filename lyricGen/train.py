import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import LyricGenModel
from dataset import Dataset

def train(dataset, model, args):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
    torch.save(model.state_dict(), "lyric_gen2.pt")
    

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--sequence-length', type=int, default=10)
    args = parser.parse_args()
    dataset = Dataset(args)
    model = LyricGenModel(dataset)
    model.load_state_dict(torch.load("lyric_gen1.pt"))
    train(dataset, model, args)
    model.load_state_dict(torch.load("lyric_gen2.pt"))
    print(predict(dataset, model, text='You'))