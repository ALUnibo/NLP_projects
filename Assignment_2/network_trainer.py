import copy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from metrics import calculate_f1_score


def get_loss_function():
    return nn.BCELoss()


def get_optimizer(model):
    return AdamW(model.parameters(), lr=0.01)


def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)


def train(model, train_dataloader, validation_dataloader, num_epochs):
    optimizer = get_optimizer(model)
    loss_function = get_loss_function()
    lr_scheduler = get_lr_scheduler(optimizer)

    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': [], 'val_class_f1': np.array([])}

    # Define your execution device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('The model will be running on', device, 'device')
    # Convert model parameters and buffers to CPU or Cuda
    model = model.to(device)

    best_model = model
    best_f1 = float('-inf')

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train(True)
        running_loss = 0.0
        i = 1

        for dl_row in train_dataloader:
            feat, lab = dl_row

            optimizer.zero_grad()

            outputs = model(feat)

            loss = loss_function(outputs, lab)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                if i != 0:
                    print('[%d, %3d] loss: %.3f' % (epoch + 1, i, running_loss / 10))
                running_loss = 0.0
            i += 1

        # evaluate the model on the validation set
        vloss, macro_f1_score = evaluate_model(model, validation_dataloader, device)
        lr_scheduler.step(vloss)

        if macro_f1_score > best_f1:
            print('New best model found! Saving it...')
            best_f1 = macro_f1_score
            best_model = copy.deepcopy(model)

    print('Finished Training')
    return best_model


def evaluate_model(model, loader, device, threshold=0.5):
    loss, outputs, labels = evaluate(model, loader, device)
    crisp_predictions = get_predictions(outputs, threshold)
    macro_f1_score, classes_f1_score = calculate_f1_score(crisp_predictions, labels)
    print(macro_f1_score, classes_f1_score)

    return loss, macro_f1_score


def evaluate(model, dataloader, device, verbose=True):
    model = model.to(device)
    outputs = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    model.eval()

    with torch.no_grad():
        loss = 0.0
        for features_i, labels_i in dataloader:
            outputs_i = model(features_i)
            # outputs_i = torch.sigmoid(outputs_i)

            outputs = torch.cat((outputs, outputs_i), 0)
            labels = torch.cat((labels, labels_i), 0)

            loss += get_loss_function()(outputs_i, labels_i).item()
    loss /= len(dataloader)
    if verbose:
        print('Loss: %.3f' % loss)
    return loss, outputs, labels


def get_predictions(outputs, threshold):
    crisp = torch.vmap(lambda x: torch.where(x > threshold, 1.0, 0.0))
    outputs = crisp(outputs)

    return outputs
