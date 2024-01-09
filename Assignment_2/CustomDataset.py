import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels, tokenizer):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # TODO: add truncation=True ???
        # features = tokenizer(features['Conclusion'].values.tolist(), padding=True, return_tensors='pt')
        conclusion_features = tokenizer(features['Conclusion'].values.tolist(), padding=True, return_tensors='pt')
        premise_features = tokenizer(features['Premise'].values.tolist(), padding=True, return_tensors='pt')
        stance_features = torch.Tensor(features['Stance']).to(device)
        features = []
        conclusion_features['input_ids'] = conclusion_features['input_ids'].to(device)
        conclusion_features['attention_mask'] = conclusion_features['attention_mask'].to(device)
        # conclusion_features['token_type_ids'] = conclusion_features['token_type_ids'].to(device)
        features.append(conclusion_features)

        premise_features['input_ids'] = premise_features['input_ids'].to(device)
        premise_features['attention_mask'] = premise_features['attention_mask'].to(device)
        # premise_features['token_type_ids'] = premise_features['token_type_ids'].to(device)
        features.append(premise_features)

        labels = torch.Tensor(labels.values.tolist())
        labels = labels.to(device)
        length = conclusion_features['input_ids'].shape[0]

        self.features = ([], [], [])
        self.labels = []

        for i in range(length):
            elem = {'input_ids': conclusion_features['input_ids'][i],
                    'attention_mask': conclusion_features['attention_mask'][i],
                    # 'token_type_ids': conclusion_features['token_type_ids'][i]
                    }
            self.features[0].append(elem)
            elem = {'input_ids': premise_features['input_ids'][i],
                    'attention_mask': premise_features['attention_mask'][i],
                    # 'token_type_ids': premise_features['token_type_ids'][i]
                    }
            self.features[1].append(elem)
            elem = stance_features[i]
            self.features[2].append(elem)
            self.labels.append(labels[i])

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, index):
        return (self.features[0][index], self.features[1][index], self.features[2][index]), self.labels[index]
