import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class WordEmbedding(nn.Module): 
    def __init__(self, vocab_size, output_dim=512):
        super(WordEmbedding, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = output_dim,
            padding_idx = 0
        )

    def forward(self, inputs) :
        x = self.embed(inputs)
        seq_len = x.size()[1]
        d_model = x.size()[2]
        pos_encoding = self.position_encoding(seq_len, d_model)
        x += pos_encoding
        return x

    def position_encoding(self,seq_len, d_model):
        """
        same dimensiton d_model as the embeddings
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        dimentions = np.arange(d_model)[np.newaxis,:]
        angles = positions/ np.power(10000, 2*(dimentions//2)/d_model)

        pos_encoding = np.zeros(angles.shape)
        pos_encoding[:,0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:,1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = torch.FloatTensor(pos_encoding)
        if torch.cuda.is_available():
            pos_encoding = pos_encoding.cuda()
        return pos_encoding

vocab_size = 30000
