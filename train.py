# Copyright: Wentao Shi, 2021
import yaml
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import dataloader_gen
from model import RNN

if __name__ == "__main__":
    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    config_dir = "./train.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # directory for results
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trained_model_dir = out_dir + 'trained_model.pt'

    # save the configuration file for future reference
    with open(out_dir + 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # training data
    dataset_dir = config['dataset_dir']
    which_vocab = config['which_vocab']
    vocab_path = config['vocab_path']
    percentage = config['percentage']

    # create dataloader
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    PADDING_IDX = config['rnn_config']['num_embeddings'] - 2
    num_workers = os.cpu_count()
    print('number of workers to load data: ', num_workers)
    print('which vocabulary to use: ', which_vocab)
    dataloader, train_size = dataloader_gen(
        dataset_dir, percentage, which_vocab,
        vocab_path, batch_size, PADDING_IDX,
        shuffle, drop_last=False
    )

    # model and training configuration
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    # create optimizer
    if config['which_optimizer'] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay, amsgrad=True
        )
    elif config['which_optimizer'] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(
            "Wrong value for optimizers! select between 'adam' and 'sgd'.")

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=10,
        cooldown=30, min_lr=0.0001,
        verbose=True
    )

    # train and validation, the results are saved.
    train_losses = []
    best_train_loss, best_train_epoch = float('inf'), None
    num_epoch = config['num_epoch']

    model.train()
    for epoch in range(1, 1 + num_epoch):
        train_loss = 0
        for data, _ in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            preds = model(data)

            # remove <sos> for targets
            targets = data[:, 1:]

            preds = preds.contiguous().view(-1, preds.size(-1))
            targets = targets.contiguous().view(-1)
            print(preds.size())
            print(targets.size())

            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()

            # accumulate loss over mini-batches
            # last minibatch's size != batch_size
            train_loss += loss.item() * data.size()[0]

        train_losses.append(train_loss / train_size)

        print('epoch {}, train loss: {}.'.format(epoch, train_losses[-1]))

        # update the saved model upon best validation loss
        if train_losses[-1] <= best_train_loss:
            best_train_epoch = epoch
            best_train_loss = train_losses[-1]
            torch.save(model.state_dict(), trained_model_dir)

        scheduler.step(train_losses[-1])

    # save train and validation losses
    with open(out_dir + 'loss.yaml', 'w') as f:
        yaml.dump(train_losses, f)
