#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import json
import numpy as np
import os
import time
import torch
from models.resnets import ResNet
from models.training import Trainer
from data.dataloaders import mnist, fashion_mnist
from plots.plots import histories_plt

def run_and_save_experiments_img(device, path_to_config):

    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "img_results_{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/config.json', 'w') as config_file:
        json.dump(config, config_file)

    num_reps = config["num_reps"]
    dataset = config["dataset"]
    model_configs = config["model_configs"]
    training_config = config["training_config"]

    results = {"dataset": dataset, "model_info": []}

    if dataset == 'mnist':
        data_loader, test_loader = mnist(256)
        output_dim = 10
    elif dataset == 'fashion_mnist':
        data_loader, test_loader = fashion_mnist(256)
        output_dim = 10

    only_success = True  # Boolean to keep track of any experiments failing

    for i, model_config in enumerate(model_configs):
        results["model_info"].append({})
        loss_histories = []
        acc_histories = []
        epoch_loss_histories = []
        epoch_acc_histories = []
        model_stats = {
            "exceeded": {"count": 0, "final_losses": []},
            "underflow": {"count": 0, "final_losses": []},
            "success": {"count": 0, "final_losses": []}
        }

        if model_config["validation"]:
            epoch_loss_val_histories = []
            epoch_acc_val_histories = []

        for j in range(num_reps):
            print("{}/{} model, {}/{} rep".format(i + 1, len(model_configs), j + 1, num_reps))
            
            model = ResNet(pow(28,2), model_config["hidden_dim"],
                               model_config["num_layers"],
                               output_dim=output_dim,
                               is_img=True)
                
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=model_config["lr"],
                                         weight_decay=0.01)

            trainer = Trainer(model, optimizer, device,
                              print_freq=training_config["print_freq"],
                              record_freq=training_config["record_freq"],
                              verbose=True,
                              save_dir=(directory, '{}_{}'.format(i, j)))

            loss_histories.append([])
            acc_histories.append([])
            epoch_loss_histories.append([])
            epoch_acc_histories.append([])

            if model_config["validation"]:
                epoch_loss_val_histories.append([])
                epoch_acc_val_histories.append([])

            # Train one epoch at a time
            for epoch in range(training_config["epochs"]):
                print("\nEpoch {}".format(epoch + 1))
                trainer.train(data_loader, 1)

                # Save info at every epoch
                loss_histories[-1] = trainer.histories['loss_history']
                acc_histories[-1] = trainer.histories['acc_history']
                epoch_loss_histories[-1] = trainer.histories['epoch_loss_history']
                epoch_acc_histories[-1] = trainer.histories['epoch_acc_history']

                if model_config["validation"]:
                    epoch_loss_val = dataset_mean_loss(trainer, test_loader, device)
                    epoch_acc_val = dataset_acc(trainer, test_loader, device)
                    if epoch == 0:
                        epoch_loss_val_histories[-1] = [epoch_loss_val]
                        epoch_acc_val_histories[-1] = [epoch_acc_val]
                    else:
                        epoch_loss_val_histories[-1].append(epoch_loss_val)
                        epoch_acc_val_histories[-1].append(epoch_acc_val)

                results["model_info"][-1]["type"] = model_config["type"]
                results["model_info"][-1]["loss_history"] = loss_histories
                results["model_info"][-1]["epoch_loss_history"] = epoch_loss_histories
                results["model_info"][-1]["acc_history"] = acc_histories
                results["model_info"][-1]["epoch_acc_history"] = epoch_acc_histories
                if model_config["validation"]:
                    results["model_info"][-1]["epoch_loss_val_history"] = epoch_loss_val_histories
                    results["model_info"][-1]["epoch_acc_val_history"] = epoch_acc_val_histories

                # Save losses and nfes at every epoch
                with open(directory + '/losses_and_nfes.json', 'w') as f:
                    json.dump(results['model_info'], f)

                # If we reached end of training, increment success counter
                if epoch == training_config["epochs"] - 1:
                    model_stats["success"]["count"] += 1

                    if len(trainer.buffer['loss']):
                        final_loss = np.mean(trainer.buffer['loss'])
                    else:
                        final_loss = None
                    model_stats["success"]["final_losses"].append(final_loss)

        # Save model stats
        with open(directory + '/model_stats{}.json'.format(i), 'w') as f:
            json.dump(model_stats, f)

    histories_plt(results["model_info"], plot_type='loss',
                  include_mean=only_success, save_fig=directory + '/losses.pdf')
    histories_plt(results["model_info"], plot_type='loss',
                  include_mean=only_success, shaded_err=True, save_fig=directory + '/losses_shaded.pdf')
    histories_plt(results["model_info"], plot_type='acc',
                  include_mean=only_success, save_fig=directory + '/accuracy.pdf')
    histories_plt(results["model_info"], plot_type='acc',
                  include_mean=only_success, shaded_err=True, save_fig=directory + '/accuracy_shaded.pdf')


def dataset_mean_loss(trainer, data_loader, device):
    """Returns mean loss of model on a dataset. 
    Useful for calculating validation loss.
    """
    epoch_loss = 0.
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred, _, __ = trainer.model(x_batch)
        loss = trainer.loss_func(y_pred, y_batch)
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def dataset_acc(trainer, data_loader, device):
    """Returns accuracy of model on a dataset. 
    Useful for calculating validation accuracy.
    """
    correct = 0
    total = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, y_pred = torch.max(trainer.model(x_batch)[0], 1)
        correct += (y_pred == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total