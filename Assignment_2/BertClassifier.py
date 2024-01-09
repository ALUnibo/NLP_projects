from transformers import BertModel
from torch import nn
import torch.functional as F
import torch


class BertClassifierC(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.embedder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.embedder.parameters():
            param.requires_grad = False
        # for param in self.embedder.pooler.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(128, n_labels)#*2+1
        self.binary = nn.Linear(768, n_labels)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = x[0]
        x = self.embedder(**x).pooler_output  # .data
        x = self.binary(x)
        x = self.sigmoid(x)
        # bin_out_1 = F.sigmoid(self.binary(x))
        # bin_out_2 = F.sigmoid(self.binary(x))
        # bin_out_3 = F.sigmoid(self.binary(x))
        # bin_out_4 = F.sigmoid(self.binary(x))
        # x = self.classifier(x)
        return x
        # return bin_out_1, bin_out_2, bin_out_3, bin_out_4  # self.softmax(x)


class BertClassifierCP(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.embedder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.embedder.parameters():
            param.requires_grad = False
        # for param in self.embedder.pooler.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(128, n_labels)#*2+1
        self.binary = nn.Linear(768*2, n_labels)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        y = x[0]
        z = x[1]
        y = self.embedder(**y).pooler_output  # .data
        z = self.embedder(**z).pooler_output # .data
        x = torch.cat([y, z], 1)
        x = self.binary(x)
        x = self.sigmoid(x)
        # bin_out_1 = F.sigmoid(self.binary(x))
        # bin_out_2 = F.sigmoid(self.binary(x))
        # bin_out_3 = F.sigmoid(self.binary(x))
        # bin_out_4 = F.sigmoid(self.binary(x))
        # x = self.classifier(x)
        return x
        # return bin_out_1, bin_out_2, bin_out_3, bin_out_4  # self.softmax(x)


class BertClassifierCPS(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.embedder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.embedder.parameters():
            param.requires_grad = False
        # for param in self.embedder.pooler.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(128, n_labels)#*2+1
        self.binary = nn.Linear(768*2+1, n_labels)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        y = x[0]
        z = x[1]
        w = x[2]
        y = self.embedder(**y).pooler_output  # .data
        z = self.embedder(**z).pooler_output  # .data
        x = torch.cat([y, z, w.reshape((8, 1))], 1)
        x = self.binary(x)
        x = self.sigmoid(x)
        # bin_out_1 = F.sigmoid(self.binary(x))
        # bin_out_2 = F.sigmoid(self.binary(x))
        # bin_out_3 = F.sigmoid(self.binary(x))
        # bin_out_4 = F.sigmoid(self.binary(x))
        # x = self.classifier(x)
        return x
        # return bin_out_1, bin_out_2, bin_out_3, bin_out_4  # self.softmax(x)
