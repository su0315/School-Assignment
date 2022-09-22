import torch
import torch.nn as nn
import torch.nn.functional as F


class SentEncoder(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(SentEncoder, self).__init__()
        """
        Fill this method. You can define and load the word embeddings here.
        You should define the convolution layer here, which use ReLU
        activation. Tips: You can use nn.Sequential to make your code cleaner.
        """
        kernel_size = 3
        stride = 1
        max_token = 50

        self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb)) 
        self.emb.requires_grad = False # when you don't train nn.Embedding during model training.
          
        # CNN with "same" padding 
        # Ensure the same embedding dimension both in input and output channel 
        self.conv1 = nn.Conv1d(pretrained_emb.shape[1], configs["conv_dim"], kernel_size, stride=stride, padding='same')
        self.conv2 = nn.Conv1d(configs["conv_dim"], configs["conv_dim"], kernel_size, stride=stride, padding='same')

    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return
        the sentence embeddings
        """
        sent = self.emb(sent)
        transposed_sent = torch.permute(sent, (0, 2, 1)) 
        # (transposed_sent.shape) = (16*300*50) 
        conv1 = F.relu(self.conv1(transposed_sent))
        conv2 = F.relu(self.conv2(conv1))
        # (conv1.shape) = (16*25*50)

        pool1 = torch.max(conv1, dim=2)[0]
        pool2 = torch.max(conv2, dim=2)[0]
        # (pool1.shape) = (16*25) 
        sent_embs = torch.concat((pool1, pool2), dim=1)
        # (sent_embs.shape) = (16*50)
        return sent_embs

class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size):
        super(NLINet, self).__init__()

        # Fill this method. You can define the FFNN, dropout and the sentence encoder here.     
        self.sent_enc = SentEncoder(configs, pretrained_emb, token_size, label_size)    
        # The size of concat pool1 and pool2 (self.sent_enc) is configs["conv_dim"]*2. 
        # The size of concat of u:v, |u-v|, u*v is configs["conv_dim"]*8.
        self.hidden = nn.Linear(configs["conv_dim"]*8, configs["mlp_hidden"]) 
        self.dropout = nn.Dropout(p=0.1)   
        self.output = nn.Linear(configs["mlp_hidden"], label_size)

    def forward(self, premise, hypothesis):
        # Fill this method. It should accept a pair of sentence (premise & hypothesis) and return the logits.  
        premise = self.sent_enc(premise)
        hypothesis = self.sent_enc(hypothesis)
        # (premise.shape) = (16*50) 
        u_v = torch.concat((premise, hypothesis), dim=1)
        subtract = torch.abs(premise-hypothesis)
        product = torch.mul(premise, hypothesis)

        concat = torch.concat((u_v, subtract, product), 1)
        # (concat.shape) = (16*200)
        dropout = self.dropout(concat)
        hidden = self.hidden(dropout)
        relu = F.relu(hidden)
        # (hidden.shape) = (16*71) 
        output = self.output(relu)
        # (output.shape)=(16*3) 
        out = F.log_softmax(output, dim=1)
        return out 
