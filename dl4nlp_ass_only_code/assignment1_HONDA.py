# -*- coding: utf-8 -*-
"""
   Deep Learning for NLP
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Remember to use PyTorch for your NN implementation.
   Original code by Hande Celikkanat & Miikka Silfverberg. Minor modifications by Sharid Loáiciga.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir


# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/) 
embeddings_file = 'GoogleNews-pruned2tweets.bin'


#--- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare! 
# You can turn in your assignment with the best settings you find.

# the output y is a 1 × 3 tensor (log pNEG, log pNEU, log pPOS).
n_classes = len(LABEL_INDICES)
n_epochs = 30 
learning_rate = 0.001
report_every = 1
verbose = False

# for the line "model"
hidden_size = 128
n_layers = 1

#--- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
  return torch.LongTensor([LABEL_INDICES[label]])

#--- model ---

class FFNN(nn.Module):
  # Feel free to add whichever arguments you like here.
  # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)
  # num_embeddings argument refers to how many elements we have in our vocabulary
  # embedding_dim is simply referring to how many dimensions we want to make the embeddings
  def __init__(self, pretrained_embeds, n_classes, hidden_size, extra_arg_1=None, extra_arg_2=None):
      super(FFNN, self).__init__()

      # WRITE CODE 
      # HERE
      self.pretrained_embeds = pretrained_embeds
      self.hidden_size = hidden_size
      self.n_classes = n_classes
      self.n_layer = n_layers

      # embedding layer
      # matrix and bias vectors associated with the hidden layer
      weights = torch.FloatTensor(pretrained_embeds)
      self.embed = nn.Embedding.from_pretrained(weights)
      self.embed.requires_grad = False # when you don't train nn.Embedding during model training 
      
      # hidden (linear) layer
      self.hidden = nn.Linear(pretrained_embeds.shape[1], hidden_size)
      # longest tweets is 31 words, we should classify tweets instead of each woords
      # we need to increase the number of input dimension around 10,000

      # output (linear) layer
      self.output = nn.Linear(hidden_size, n_classes)

      

  def forward(self, x):
      # WRITE CODE 
      x = self.embed(x)
      # shape of x is (13, 300) so 300 are the embedding dimention, the other one is each tweet length 
      # sum up the embed of the weights of each words in a tweet
      x = torch.sum(x, dim=0) 
      
      x = self.hidden(x)
      x = F.relu(x)
      x = self.output(x)
      output = F.log_softmax(x, dim = 0)
      return output
      
#--- "main" ---

if __name__=='__main__':
  #--- data loading ---
  data = read_semeval_datasets(data_dir)
  gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file), binary=True)
  pretrained_embeds = gensim_embeds.vectors
  # To convert words in the input tweet to indices of the embeddings matrix:
  word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}

  #--- set up ---
  # WRITE CODE HERE
  model = FFNN(pretrained_embeds, n_classes, hidden_size)
  loss_function = torch.nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr = learning_rate)
  

  #--- training ---
  for epoch in range(n_epochs):
    total_loss = 0
    for tweet in data['training']: # loop over ID, Sentiment, BODY 
      optimizer.zero_grad() 
      gold_class = label_to_idx(tweet['SENTIMENT'])
      # WRITE CODE HERE
      tweet_index_list = []
      for word in tweet['BODY']: #loop over the word in tweet
        #(word_to_idx) is Dictionary (word : idx)
        #(word_to_idx[word])# idx(value) of the word(key) 
        # if word does not exists in the word_to_idx, pass
        if word not in word_to_idx: # when we want to check the key of dictionary, we just type the dic name
          pass
        else:
          
          tweet_index_list.append(word_to_idx[word])  
      
      tweet_tensor = torch.LongTensor(tweet_index_list) # Put the word in Twitter weight into torch Tensor
      prediction = model (tweet_tensor) # call the model with the tweet_tensor # we do not need to call forward ()
      loss = loss_function(prediction.view(1, -1), gold_class) 
      total_loss += loss
      loss.backward()
      optimizer.step()

    if ((epoch+1) % report_every) == 0:
      print('epoch: %d, loss: %.4f' % (epoch, total_loss*100/len(data['training'])))


  # Feel free to use the development data to tune hyperparameters if you like!

  #--- test ---
  correct = 0
  with torch.no_grad():
    for tweet in data['test.gold']:
      gold_class = label_to_idx(tweet['SENTIMENT'])
    
      # WRITE CODE HERE
      tweet_index_list = []
      for word in tweet['BODY']: #loop over the word in tweet
        if word not in word_to_idx: # when we want to check the key of dictionary, we just type the dic name
          pass
        else:
          tweet_index_list.append(word_to_idx[word])  

      tweet_tensor = torch.LongTensor(tweet_index_list)

      prediction = model(tweet_tensor)
      
      predicted =torch.argmax(prediction)
      correct += torch.eq(predicted,gold_class).item()

      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], predicted))
        
    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))




