import torch
from torcheval.metrics.functional import binary_f1_score
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score


def get_loss_function():
    return nn.BCELoss()


def get_optimizer(model):
    return Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def train(model, train_dataloader, validation_dataloader, num_epochs):
    best_accuracy = 0.0

    optimizer = get_optimizer(model)
    loss_function = get_loss_function()

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model = model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        i = 0

        for dl_row in train_dataloader:
            feat, lab = dl_row

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using data from the training set
            outputs = model(feat)

            # compute the loss based on model output and real labels
            loss = loss_function(outputs, lab)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item()  # extract the loss value
            if i % 10 == 0:
                if i != 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                # zero the loss
                running_loss = 0.0
            i += 1

        # evaluate the model on the validation set
        accuracy = evaluate(model, validation_dataloader)
        print('Accuracy of the network on the validation set: %.3f' % accuracy)

        # we want to save the model if the accuracy is the best
        # if accuracy > best_accuracy:
        #     saveModel()
        #     best_accuracy = accuracy


def f1_score(predictions, targets):
    cols = predictions.shape[1]
    per_col_scores = torch.zeros(cols)
    for i in range(cols):
        per_col_scores[i] = binary_f1_score(predictions[:, i], targets[:, i])
    return torch.mean(per_col_scores)


def evaluate(model, validation_dataloader, threshold=0.7):
    model.eval()
    validation_data = next(iter(validation_dataloader))
    assert validation_data[0]['input_ids'].shape[0] == len(validation_dataloader.dataset)
    features, labels = validation_data
    outputs = model(features)

    crisp = torch.vmap(lambda x: torch.where(x > threshold, 1.0, 0.0))
    outputs = crisp(outputs)
    return f1_score(outputs, labels)
