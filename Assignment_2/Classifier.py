from transformers import BertModel, AutoModel, RobertaModel
from torch import nn
import torch


class ClassifierC(nn.Module):
    def __init__(self, name, head_size):
        super(ClassifierC, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(head_size, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x[0]
        attention_mask = x['attention_mask'].unsqueeze(-1)
        x = self.embedder(**x).last_hidden_state
        x = x * attention_mask
        x = x.mean(dim=1)
        x = self.linear(x)
        x = (self.tanh(x) + 1) / 2
        return x


class ClassifierCP(nn.Module):
    def __init__(self, name, head_size):
        super(ClassifierCP, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(head_size * 2, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = x[0]
        z = x[1]
        attention_mask_y = y['attention_mask'].unsqueeze(-1)
        attention_mask_z = z['attention_mask'].unsqueeze(-1)
        y = self.embedder(**y).last_hidden_state
        z = self.embedder(**z).last_hidden_state
        y = y * attention_mask_y
        z = z * attention_mask_z
        y = y.mean(dim=1)
        z = z.mean(dim=1)
        x = torch.cat([y, z], 1)
        x = self.linear(x)
        x = (self.tanh(x) + 1) / 2
        return x


class ClassifierCPS(nn.Module):
    def __init__(self, name, head_size):
        super(ClassifierCPS, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(head_size * 2 + 1, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = x[0]
        z = x[1]
        w = x[2]
        attention_mask_y = y['attention_mask'].unsqueeze(-1)
        attention_mask_z = z['attention_mask'].unsqueeze(-1)
        y = self.embedder(**y).last_hidden_state
        z = self.embedder(**z).last_hidden_state
        y = y * attention_mask_y
        z = z * attention_mask_z
        y = y.mean(dim=1)
        z = z.mean(dim=1)
        x = torch.cat([y, z, w.reshape((y.shape[0], 1))], 1)
        x = self.linear(x)
        x = (self.tanh(x) + 1) / 2
        return x
