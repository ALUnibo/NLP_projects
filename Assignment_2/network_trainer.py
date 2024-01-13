import copy
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup

from metrics import calculate_f1_score


def get_loss_function():
    return nn.BCELoss()


def get_optimizer(model):
    return AdamW(model.parameters(), lr=0.01)


def get_lr_scheduler(optimizer, num_training_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


def train(model, train_dataloader, validation_dataloader, num_epochs):
    optimizer = get_optimizer(model)
    loss_function = get_loss_function()
    lr_scheduler = get_lr_scheduler(optimizer, num_epochs)

    history = {'train_loss': [],
               'val_loss': [],
               'val_macro_f1': [],
               'val_class_f1': [],
               'best_val_macro_f1': float('-inf')}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('The model will be running on', device, 'device')
    # Convert model parameters and buffers to CPU or Cuda
    model = model.to(device)

    best_model = model

    for epoch in range(num_epochs):
        print('Epoch %d/%d on model %s:' % (epoch + 1, num_epochs, model.name_))
        model.train(True)
        running_loss = 0.0
        train_loss = 0.0
        i = 0

        for dl_row in train_dataloader:
            feat, lab = dl_row

            optimizer.zero_grad()

            outputs = model(feat)

            loss = loss_function(outputs, lab)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            train_loss += loss.item()

            i += 1
            if i % 10 == 0:
                print('[%d, %3d] loss: %.3f' % (epoch + 1, i, running_loss / 10))
                running_loss = 0.0

        train_loss /= i
        history['train_loss'].append(train_loss)

        # evaluate the model on the validation set
        print('Evaluating the model on the validation set...')
        vloss, macro_f1_score, class_f1_score, _, _, _ = evaluate_model(model, validation_dataloader, device)
        history['val_loss'].append(vloss)
        history['val_macro_f1'].append(macro_f1_score)
        history['val_class_f1'].append(class_f1_score)

        lr_scheduler.step()

        if macro_f1_score > history['best_val_macro_f1']:
            print('New best model found! Saving it...')
            history['best_val_macro_f1'] = macro_f1_score
            best_model = copy.deepcopy(model)

    history['val_class_f1'] = np.array(history['val_class_f1'])

    print('Finished Training\n')
    return best_model, history


def evaluate_model(model, loader, device, threshold=0.5, verbose=True):
    loss, outputs, labels = evaluate(model, loader, device, verbose)
    crisp_predictions = get_predictions(outputs, threshold)
    macro_f1_score, class_f1_score = calculate_f1_score(crisp_predictions, labels)
    macro_f1_score, class_f1_score = macro_f1_score.item(), class_f1_score.cpu().numpy()
    if verbose:
        print('Macro F1 score: %.3f - Class F1 score: %s' % (macro_f1_score, np.array_str(class_f1_score, precision=3)))

    return loss, macro_f1_score, class_f1_score, outputs.cpu().numpy(), labels.cpu().numpy(), \
        crisp_predictions.cpu().numpy()


def evaluate(model, dataloader, device, verbose=True):
    model = model.to(device)
    outputs = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    model.eval()

    with torch.no_grad():
        loss = 0.0
        for features_i, labels_i in dataloader:
            outputs_i = model(features_i)

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
