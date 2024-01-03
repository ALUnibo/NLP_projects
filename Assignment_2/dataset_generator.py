import torch
from datasets import Dataset


def generate_datasets(tokenizer, training_set, validation_set, test_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = Dataset.from_pandas(training_set).with_format('torch', device=device)
    val_data = Dataset.from_pandas(validation_set).with_format('torch', device=device)
    test_data = Dataset.from_pandas(test_set).with_format('torch', device=device)

    train_data = train_data.map(lambda x: tokenizer(x['Conclusion'], truncation=True), batched=True)
    train_data = train_data.map(lambda x: tokenizer(x['Premise'], truncation=True), batched=True)

    val_data = val_data.map(lambda x: tokenizer(x['Conclusion'], truncation=True), batched=True)
    val_data = val_data.map(lambda x: tokenizer(x['Premise'], truncation=True), batched=True)

    test_data = test_data.map(lambda x: tokenizer(x['Conclusion'], truncation=True), batched=True)
    test_data = test_data.map(lambda x: tokenizer(x['Premise'], truncation=True), batched=True)

    return train_data, val_data, test_data
