# -*- coding: utf-8 -*-
"""
    Deep Learning for NLP
    Assignment 3: Language Identification using Recurrent Architectures
    Based on original code by Hande Celikkanat & Miikka Silfverberg
"""
from random import choice, random, shuffle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk

from data import read_datasets, WORD_BOUNDARY, UNK, HISTORY_SIZE
from paths import data_dir

torch.set_num_threads(10)

#--- hyperparameters ---
N_EPOCHS = 10
LEARNING_RATE = 3e-4
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
BATCH_SIZE = 30
N_LAYERS = 1


#--- models ---
class LSTMModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # WRITE CODE HERE
        self.embedding = torch.nn.Embedding(self.character_set_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers)
        self.output = torch.nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, inputs):
        # WRITE CODE HERE
        embeds = self.embedding(inputs)
        # Recommendation: use a single input for lstm layer (no special initialization of the hidden layer):
        lstm_out, hidden = self.lstm(embeds)

        # WRITE MORE CODE HERE
        # reshape from 3 dimension to 2 dimension using view
        output = self.output(hidden[0].view(BATCH_SIZE,-1)) # hidden[0] is hidden, because hidden is (hidden, cell) # lstm[-1] works too
        prob = F.log_softmax(output, dim=1) 
        return prob

class GRUModel(nn.Module): 
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(GRUModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # WRITE CODE HERE
        self.embedding = torch.nn.Embedding(self.character_set_size, self.embedding_dim)
        self.gru = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers)
        self.output = torch.nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, inputs):
        # WRITE CODE HERE
        embeds = self.embedding(inputs)
        # Recommendation: use a single input for gru layer (no special initialization of the hidden layer):
        gru_out, hidden = self.gru(embeds)
        
        # WRITE MORE CODE HERE
        output = self.output(hidden[0].view(BATCH_SIZE,-1))
        prob = F.log_softmax(output, dim=1) 
        return prob
        

class RNNModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(RNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # WRITE CODE HERE
        self.embedding = torch.nn.Embedding(self.character_set_size, self.embedding_dim)
        self.rnn = torch.nn.RNN(self.embedding_dim, self.hidden_dim, self.n_layers)
        self.output = torch.nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, inputs):
        # WRITE CODE HERE
        embeds = self.embedding(inputs)
        # Recommendation: use a single input for rnn layer (no special initialization of the hidden layer):
        rnn_out, hidden = self.rnn(embeds)
        
        # WRITE MORE CODE HERE
        output = self.output(hidden[0].view(BATCH_SIZE,-1))
        prob = F.log_softmax(output, dim=1) 
        return prob

# --- auxilary functions ---
def get_minibatch(minibatchwords, character_map, languages):
    # WRITE CODE HERE
    sequence_list = []
    mb_y_list = []

    for batch in minibatchwords:
        sequence = batch['TENSOR']
        sequence_list.append(sequence)
        y = label_to_idx(batch['LANGUAGE'],languages)
        mb_y_list.append(y)

    from torch.nn.utils.rnn import pad_sequence
    mb_x = pad_sequence(sequence_list, padding_value = character_map['#']) 
    mb_y = torch.as_tensor(mb_y_list)

    return  mb_x,  mb_y

def label_to_idx(lan, languages):
    languages_ordered = list(languages)
    languages_ordered.sort()
    return torch.LongTensor([languages_ordered.index(lan)])

def get_word_length(word_ex):
    return len(word_ex['WORD'])

def evaluate(dataset,model,eval_batch_size,character_map,languages):
    correct = 0
    
    # WRITE CODE HERE IF YOU LIKE
    for i in range(0,len(dataset),eval_batch_size):
        minibatchwords = dataset[i:i+eval_batch_size]    
        mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)

        # WRITE CODE HERE    
        prediction = model(mb_x)
        predicted =torch.argmax(prediction, dim = 1)
        correct += torch.sum(torch.eq(predicted,mb_y))

    return correct * 100.0 / len(dataset)

if __name__=='__main__':

    # --- select the recurrent layer according to user input ---
    if len(sys.argv) < 2:
        print('-------')
        print('You didn''t provide any arguments!')
        print('Using LSTM model as default')
        print('To select a model, call the program with one of the arguments: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('-------')
        model_choice = 'lstm'
    elif len(sys.argv) == 2:
        print('-------')
        print('Running with ' + sys.argv[1][1:] + ' model')
        print('-------')        
        model_choice = sys.argv[1][1:]
    else:
        print('-------')
        print('Wrong number of arguments')
        print('Please call the model with exactly one argument, which can be: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('Using LSTM model as default')
        print('-------')        
        model_choice = 'lstm'

 
    #--- initialization ---
    if BATCH_SIZE == 1:
        data, character_map, languages = read_datasets('uralic.mini',data_dir)
    else:
        data, character_map, languages = read_datasets('uralic',data_dir)

    trainset = [datapoint for lan in languages for datapoint in data['training'][lan]]
    n_languages = len(languages)
    character_set_size = len(character_map)

    model = None
    if model_choice == 'lstm':
        model = LSTMModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'gru':
        model = GRUModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'rnn':
        model = RNNModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)


    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_function = nn.NLLLoss()

    # --- training loop ---
    for epoch in range(N_EPOCHS):
        total_loss = 0

        # Generally speaking, it's a good idea to shuffle your
        # datasets once every epoch.
        shuffle(trainset)

        # WRITE CODE HERE
        # Sort your training set according to word-length, 
        # so that similar-length words end up near each other
        # You can use the function get_word_length as your sort key.
        trainset.sort(key=get_word_length)

        for i in range(0,len(trainset),BATCH_SIZE):
            minibatchwords = trainset[i:i+BATCH_SIZE]

            mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)
            # WRITE CODE HERE
            optimizer.zero_grad() 
            prob = model(mb_x)
            loss = loss_function(prob, mb_y)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            train_acc = evaluate(trainset,model,BATCH_SIZE,character_map,languages)
            dev_acc = evaluate(data['dev'],model,BATCH_SIZE,character_map,languages)
            print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
                  (epoch+1, total_loss, train_acc, dev_acc))
    
    # --- test ---    
    test_acc = evaluate(data['test'],model,BATCH_SIZE,character_map,languages)        
    print('test acc: %.2f%%' % (test_acc))