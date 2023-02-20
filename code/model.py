import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""

class GRU(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, batch_size, out_size, pretrained_vec=None, emb_update=True):
        super(GRU, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.batch_size = batch_size
        self.out_size = out_size

        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        if pretrained_vec:
            # Load pretrained vectors
            self.emb.weight.data.copy_(pretrained_vec)
        if not emb_update:
            # Freeze embedding layer
            self.emb.weight.requires_grad = False
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.emb_size, self.hid_size)
        self.fc = nn.Linear(self.hid_size, self.out_size)

    def forward(self, inputs, lengths, device):
        self.hidden = self.init_hidden(device)
        x = self.emb(inputs)
        x = pack_padded_sequence(x, lengths)
        gru_out, self.hidden = self.gru(x, self.hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        out = gru_out[-1, :, :]
        out = self.dropout(out)
        out = self.fc(out)

        return F.log_softmax(out, dim=-1)


    def init_hidden(self, device):
        return torch.zeros(1, self.batch_size, self.hid_size).to(device)
