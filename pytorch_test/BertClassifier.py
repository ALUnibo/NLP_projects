from transformers import BertModel
from torch import nn
import torch.functional as F


class BertClassifier(nn.Module):
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
        x = self.embedder(**x).pooler_output.data
        x = self.binary(x)
        x = self.sigmoid(x)
        # bin_out_1 = F.sigmoid(self.binary(x))
        # bin_out_2 = F.sigmoid(self.binary(x))
        # bin_out_3 = F.sigmoid(self.binary(x))
        # bin_out_4 = F.sigmoid(self.binary(x))
        # x = self.classifier(x)
        return x
        # return bin_out_1, bin_out_2, bin_out_3, bin_out_4  # self.softmax(x)
