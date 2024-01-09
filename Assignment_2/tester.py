import torch
from metrics import f1_score


def test(model, validation_dataloader, threshold=0.7):
    model.eval()
    validation_data = next(iter(validation_dataloader))
    assert validation_data[0]['input_ids'].shape[0] == len(validation_dataloader.dataset)
    features, labels = validation_data
    outputs = model(features)

    crisp = torch.vmap(lambda x: torch.where(x > threshold, 1.0, 0.0))
    outputs = crisp(outputs)
    return f1_score(outputs, labels)
