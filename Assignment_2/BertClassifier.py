from transformers import BertModel, AutoModel, RobertaModel
from torch import nn
import torch


class BertClassifierC(nn.Module):
    def __init__(self, name, head_size):
        super(BertClassifierC, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        # for param in self.embedder.pooler.parameters():
        #     param.requires_grad = True
        self.linear = nn.Linear(head_size, 4)

    def forward(self, x):
        x = x[0]
        x = self.embedder(**x).pooler_output
        x = self.linear(x)
        return x


class BertClassifierCP(nn.Module):
    def __init__(self, name, head_size):
        super(BertClassifierCP, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(head_size * 2, 4)

    def forward(self, x):
        y = x[0]
        z = x[1]
        y = self.embedder(**y).pooler_output
        z = self.embedder(**z).pooler_output
        x = torch.cat([y, z], 1)
        x = self.linear(x)
        return x


class BertClassifierCPS(nn.Module):
    def __init__(self, name, head_size):
        super(BertClassifierCPS, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(head_size * 2 + 1, 256)
        self.linear2 = nn.Linear(256, 4)

    def forward(self, x):
        y = x[0]
        z = x[1]
        w = x[2]
        y = self.embedder(**y).pooler_output
        z = self.embedder(**z).pooler_output
        x = torch.cat([y, z, w.reshape((y.shape[0], 1))], 1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
