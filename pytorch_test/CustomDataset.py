import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels, tokenizer):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: add truncation=True ???
        features = tokenizer(features['Conclusion'].values.tolist(), padding=True, return_tensors='pt')
        features['input_ids'] = features['input_ids'].to(device)
        features['attention_mask'] = features['attention_mask'].to(device)
        features['token_type_ids'] = features['token_type_ids'].to(device)

        labels = torch.Tensor(labels.values.tolist())
        labels = labels.to(device)
        length = features['input_ids'].shape[0]

        self.features = []
        self.labels = []

        for i in range(length):
            elem = {'input_ids': features['input_ids'][i], 'attention_mask': features['attention_mask'][i],
                    'token_type_ids': features['token_type_ids'][i]}
            self.features.append(elem)
            self.labels.append(labels[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
