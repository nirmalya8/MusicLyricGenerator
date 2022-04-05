import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import get_preprocessed_data
from model import RapLyricGen

num_hidden = 256
num_layers = 4
embed_size = 200
drop_prob = 0.3
lr = 0.001
num_epochs = 50
batch_size = 32

x_idx,y_idx, vocab_size,_,_ = get_preprocessed_data()
model = RapLyricGen(num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()
#model.train()
model.load_state_dict(torch.load("rap_gen_2.pt"))
def get_next_batch(x, y, batch_size):
    for itr in range(batch_size, x.shape[0], batch_size):
        batch_x = x[itr - batch_size:itr, :]
        batch_y = y[itr - batch_size:itr, :]
        yield batch_x, batch_y


for epoch in range(num_epochs):

    # initialize hidden state
    hidden_layer = model.init_hidden(batch_size)
    i = 0  
    for x, y in get_next_batch(x_idx, y_idx, batch_size):
        # convert numpy arrays to PyTorch arrays
        inputs = torch.from_numpy(x).type(torch.LongTensor)
        act = torch.from_numpy(y).type(torch.LongTensor)

        # reformat the hidden layer
        hidden_layer = tuple([layer.data for layer in hidden_layer])
        

        # obtain the zero-accumulated gradients from the model
        model.zero_grad()
            
        # get the output from the model
        output, hidden = model(inputs, hidden_layer)
            
        # calculate the loss from this prediction
        loss = loss_func(output, act.view(-1))
        if i%10 == 0:
            print({ 'epoch': epoch, 'batch': i, 'loss': loss.item() }) 
        i+=1  

        # back-propagate to update the model
        loss.backward()

        # prevent exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        # update weigths using the optimizer
        optimizer.step()     

torch.save(model.state_dict(), "rap_gen_3.pt")