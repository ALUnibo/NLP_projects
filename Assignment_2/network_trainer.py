import torch
from torcheval.metrics.functional import binary_f1_score
from torch import nn
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup


def get_loss_function():
    return nn.BCEWithLogitsLoss()


def get_optimizer(model):
    return Adam(model.parameters(), lr=2e-05)


def get_lr_scheduler(optimizer, num_training_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


def train(model, train_dataloader, validation_dataloader, num_epochs):
    optimizer = get_optimizer(model)
    loss_function = get_loss_function()
    lr_scheduler = get_lr_scheduler(optimizer, len(train_dataloader) * num_epochs)

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The model will be running on', device, 'device')
    # Convert model parameters and buffers to CPU or Cuda
    model = model.to(device)

    evaluate(model, validation_dataloader)

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
            lr_scheduler.step()

            running_loss += loss.item()
            if i % 10 == 0:
                if i != 0:
                    print('[%d, %3d] loss: %.3f' % (epoch + 1, i, running_loss / 10))
                running_loss = 0.0
            i += 1

        # evaluate the model on the validation set
        score = evaluate(model, validation_dataloader)

        # we want to save the model if the accuracy is the best
        # if accuracy > best_accuracy:
        #     saveModel()
        #     best_accuracy = accuracy


def f1_score(predictions, targets, verbose=False):
    cols = predictions.shape[1]
    single_class_scores = torch.zeros(cols)
    for i in range(cols):
        single_class_scores[i] = binary_f1_score(predictions[:, i], targets[:, i])
        if verbose:
            print('F1 score for column %d: %.3f' % (i, single_class_scores[i]))
    return single_class_scores


def evaluate(model, validation_dataloader):
    model.eval()

    with torch.no_grad():
        vloss = 0.0
        for features, labels in validation_dataloader:
            outputs = model(features)
            outputs = torch.sigmoid(outputs)
            vloss += get_loss_function()(outputs, labels).item()
        vloss /= len(validation_dataloader)
        print('Validation loss: %.3f' % vloss)

    all_thresholds = torch.arange(0.5, 0.8, 0.05)
    all_f1_scores = torch.zeros(len(all_thresholds))
    best_f1_score = float('-inf')

    for i, t in enumerate(all_thresholds):
        crisp = torch.vmap(lambda x: torch.where(x > t, 1.0, 0.0))
        outputs_i = crisp(outputs)
        f1_scores_i = f1_score(outputs_i, labels)
        all_f1_scores[i] = torch.mean(f1_scores_i)
        if all_f1_scores[i] > best_f1_score:
            best_f1_score = all_f1_scores[i]
            single_class_scores = f1_scores_i
            best_threshold = t

    for i in range(len(single_class_scores)):
        print('F1 score for column %d: %.3f' % (i, single_class_scores[i]))

    print('F1 score of the network on the validation set: %.3f, with threshold: %.3f' % (best_f1_score, best_threshold))

    return best_f1_score, best_threshold
