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
from models.neural_odes import *
from models.training import Trainer
from data.dataloaders import *
from plots.plots import histories_plt


def run_experiments(device, 
                    data_dim=2, 
                    viz_batch_size=512, 
                    num_reps=5,
                    datasets=[], 
                    model_configs=[], 
                    training_config={}):
    """What about
  
    Args:
        arg: what it is
    
    Returns:
        what returns
    """
    """
    Runs experiments for various model configurations on various datasets.
    """
    results = []
    for dataset in datasets:
        if dataset["type"] == "sphere":
            data_object = ConcentricSphere(data_dim,
                                           dataset["inner_range"],
                                           dataset["outer_range"],
                                           dataset["num_points_inner"],
                                           dataset["num_points_outer"])

        
        train_size = int(0.8 * len(data_object))
        test_size = len(data_object) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data_object, [train_size, test_size])

        data_loader = DataLoader(train_dataset,
                                 batch_size=training_config["batch_size"],
                                 shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=training_config["batch_size"],
                                 shuffle=True)

        results.append({"dataset": dataset, "model_info": [], "tensors": [],
                        "models": []})

        # Retrieve inputs and targets which will be used to visualize how models
        # transform inputs to features
        data_loader_viz = DataLoader(data_object,
                                     batch_size=viz_batch_size,
                                     shuffle=True)
        for batch in data_loader_viz:
            break
        inputs, targets = batch

        for i, model_config in enumerate(model_configs):
            # Check whether model is ODE based or a ResNet
            is_ode = model_config["type"] == "odenet" or model_config["type"] == "anode"

            # Initialize histories
            loss_histories = []
            epoch_loss_histories = []
            acc_histories = []
            epoch_acc_histories = []
            features = []
            predictions = []
            models = []
            start = time.time()
            epoch_loss_val_histories = []
            epoch_acc_val_histories = []

            for j in range(num_reps):
                print("{}/{} model, {}/{} rep".format(i + 1, len(model_configs), j + 1, num_reps))

                if is_ode:
                    if model_config["type"] == "odenet":
                        augment_dim = 0
                    else:
                        augment_dim = model_config["augment_dim"]

                    model = NeuralODE(device, data_dim, model_config["hidden_dim"],
                                   augment_dim=augment_dim)
                else:
                    model = ResNet(data_dim, model_config["hidden_dim"],
                                   model_config["num_layers"])

                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=model_config["lr"], weight_decay=0.01)

                trainer = Trainer(model, optimizer, device,
                                  print_freq=training_config["print_freq"],
                                  record_freq=training_config["record_freq"],
                                  verbose=False)

                trainer.train(data_loader, training_config["epochs"])

                loss_histories.append(trainer.histories["loss_history"])
                epoch_loss_histories.append(trainer.histories["epoch_loss_history"])
                acc_histories.append(trainer.histories['acc_history'])
                epoch_acc_histories.append(trainer.histories['epoch_acc_history'])

                epoch_loss_val = dataset_mean_loss(trainer, test_loader, device)
                epoch_acc_val = dataset_acc(trainer, test_loader, device, "none")
                epoch_loss_val_histories.append(epoch_loss_val)
                epoch_acc_val_histories.append(epoch_acc_val)

                # Add trained model
                models.append(model)

            results[-1]["model_info"].append({
                "type": model_config["type"],
                "loss_history":  loss_histories,
                "epoch_loss_history": epoch_loss_histories,
                "avg_time": (time.time() - start) / num_reps,
                "acc_history":  acc_histories,
                "epoch_acc_history": epoch_acc_histories,
                "epoch_loss_val_history":   epoch_loss_val_histories,
                "epoch_acc_val_history":    epoch_acc_val_histories
            })

            results[-1]["models"].append(models)

    return results

def dataset_mean_loss(trainer, data_loader, device):
    """What about
  
    Args:
        arg: what it is
    
    Returns:
        what returns
    """
    """
    Returns mean loss of model on a dataset. Useful for calculating validation loss.
    """
    epoch_loss = 0.
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred, _ = trainer.model(x_batch)
        loss = trainer.loss_func(y_pred, y_batch)
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def dataset_acc(trainer, 
                data_loader, 
                device, 
                noise_type="none", 
                noise_param=0):
    """What about
  
    Args:
        arg: what it is
    
    Returns:
        what returns
    """
    """Returns accuracy of model on a dataset. Useful for calculating
    validation accuracy.
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


def run_experiments_from_config(device, path_to_config):
    """
    Runs an experiment from a config file.
    """
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    results = run_experiments(device, data_dim=config["data_dim"],
                              viz_batch_size=config["viz_batch_size"],
                              num_reps=config["num_reps"],
                              datasets=config["datasets"],
                              model_configs=config["model_configs"],
                              training_config=config["training_config"])

    return results


def run_and_save_experiments(device, 
                                path_to_config, 
                                save_models=False,
                                save_tensors=False):
    """What about
  
    Args:
        arg: what it is
    
    Returns:
        what returns
    """
    """
    Runs an experiment from a config file, saves logs and generates various
    plots of results.
    """
    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "results_{}".format(timestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as config_file:
        json.dump(config, config_file)

    # Run experiments
    results = run_experiments_from_config(device, path_to_config)

    # Create figures and save experiments
    for i in range(len(results)):
        # Create directory to store result
        subdir = directory + '/{}'.format(i)
        os.makedirs(subdir)
        # Save dataset information
        with open(subdir + '/dataset.json', 'w') as f:
            json.dump(results[i]['dataset'], f)

        # Save model and losses info
        with open(subdir + '/model_losses.json', 'w') as f:
            json.dump(results[i]['model_info'], f)

        # Create losses figure for this dataset
        histories_plt(results[i]["model_info"], plot_type='loss',
                      save_fig=subdir + '/losses.pdf')
        histories_plt(results[i]["model_info"], plot_type='loss',
                      shaded_err=True, save_fig=subdir + '/losses_shaded.pdf')

        # For each individual run, save model and save input-feature figures
        for j in range(len(results[i]["model_info"])):
            model_type = results[i]["model_info"][j]["type"]
            models = results[i]["models"][j]

            if save_models:
                for k in range(len(models)):
                    torch.save(models[k], subdir + '/model_{}_{}_{}.pt'.format(model_type, j, k))