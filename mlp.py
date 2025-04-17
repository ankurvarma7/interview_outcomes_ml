import features
import matplotlib.pyplot as plt
import numpy as np
import test_train_sets
import time
import torch
import torch.nn as nn
import torch.optim as optim


# Packages the hyperparameters to the problem in one class.
# The MLP model will have as many hidden layers as specified by the length of hidden_layer_sizes
# hyperparameter, and each hidden layer will the corresponding size.
class MLPHyperParameters:
    def __init__(
        self,
        k,
        summarize_bert_features,
        learning_rate,
        num_epochs,
        hidden_layer_sizes,
        dropout_probability,
    ):
        self.k = k
        self.summarize_bert_features = summarize_bert_features
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_probability = dropout_probability
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


# Does one full iteration of training over the data in training_data_loader, updating
# the model using the given optimizer and loss_function.
def train_one_epoch(training_data_loader, model, optimizer, loss_function):
    model.train()
    all_losses = np.array([])
    for features, labels in training_data_loader:
        optimizer.zero_grad()
        predictions = model(features)
        # print("Predictions: ", predictions)
        # print("Labels: ", labels)
        loss = loss_function(predictions, labels)
        all_losses = np.append(all_losses, loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(all_losses)


# Computes the mean loss with respect to the given loss function over the data in the
# given data loader. The `flatten` parameter is used to control the shape of the
# inputs and outputs.
def mean_validation_loss(validation_data_loader, model, loss_function):
    model.eval()
    all_validation_losses = np.array([])
    for validation_samples, validation_labels in validation_data_loader:
        predictions = model(validation_samples)
        loss = loss_function(predictions, validation_labels).item()
        all_validation_losses = np.append(all_validation_losses, loss)
    return np.mean(all_validation_losses)


# Plots the learning curve (validation loss over epochs).
def plot_losses(train_losses_by_epoch, validation_losses_by_epoch):
    plt.scatter(
        range(len(train_losses_by_epoch)),
        train_losses_by_epoch,
        s=9,
        label="Training loss (dropout applied)",
    )
    plt.scatter(
        range(len(validation_losses_by_epoch)),
        validation_losses_by_epoch,
        s=9,
        label="Validation loss (no dropout)",
    )
    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()


# Trains the given model against the given data, with given hyperparameters, evaluating
# against the given validation data. Returns a list of losses against both the training
# data and validation data.
def train_and_report(
    model,
    training_data_loader,
    validation_data_loader,
    num_epochs,
    learning_rate,
):
    train_losses_by_epoch = []
    validation_losses_by_epoch = []
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0
    )
    loss_function = nn.MSELoss()
    start_time = time.time()
    for epoch in range(num_epochs):
        train_losses_by_epoch.append(
            train_one_epoch(training_data_loader, model, optimizer, loss_function)
        )
        validation_losses_by_epoch.append(
            mean_validation_loss(validation_data_loader, model, loss_function)
        )
        if epoch % 10 == 9:
            print("Trained to epoch: ", epoch)
    end_time = time.time()
    print("Time to train (sec): ", end_time - start_time)
    print("Latest training loss: ", train_losses_by_epoch[-1])
    print("Latest validation loss: ", validation_losses_by_epoch[-1])
    return train_losses_by_epoch, validation_losses_by_epoch


# A feedforward net with leaky ReLU activations. If `dropout` is true, then dropout layers will be added between
# all intermediate layers.
def ff_leaky_relu_model(hidden_layer_sizes, input_size, dropout_probability):
    args = [nn.Linear(input_size, hidden_layer_sizes[0]), nn.LeakyReLU()]
    if dropout_probability > 0:
        args.append(nn.Dropout(dropout_probability))
    for i in range(len(hidden_layer_sizes) - 1):
        args += [
            nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]),
            nn.LeakyReLU(),
        ]
        if dropout_probability > 0:
            args.append(nn.Dropout(dropout_probability))
    args.append(nn.Linear(hidden_layer_sizes[-1], 2))
    return nn.Sequential(*args)


# For the given hyperparameters, runs cross-validation and plots an averaged learning curve
# across all validation runs.
def evaluate_with_hyperparameters(hps: MLPHyperParameters):
    training_losses = []
    validation_losses = []
    for validation_index in [0, 1, 2, 3]:
        validation_participants = test_train_sets.get_train_set(validation_index)
        training_participants = [
            participant
            for participant_set in [
                test_train_sets.get_train_set(i)
                for i in [0, 1, 2, 3]
                if i != validation_index
            ]
            for participant in participant_set
        ]
        training_data_loader = torch.utils.data.DataLoader(
            features.dataset_for_participants(
                training_participants, False, hps.summarize_bert_features, hps.k
            )
        )
        validation_data_loader = torch.utils.data.DataLoader(
            features.dataset_for_participants(
                validation_participants, False, hps.summarize_bert_features, hps.k
            )
        )
        input_size = next(iter(training_data_loader))[0].shape[1]
        model = ff_leaky_relu_model(
            hps.hidden_layer_sizes, input_size, hps.dropout_probability
        )
        training_loss, validation_loss = train_and_report(
            model,
            training_data_loader,
            validation_data_loader,
            hps.num_epochs,
            hps.learning_rate,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    plot_losses(
        np.mean(np.vstack(training_losses), axis=0),
        np.mean(np.vstack(validation_losses), axis=0),
    )
    return np.mean(np.mean(np.vstack(validation_losses), axis=0)[-5:])


# The start of hyperparameter optimzation. Reasonable guesses for everything; we'll start
# by tuning the learning rate.
def initial_hps_guess_with_learning_rate(learning_rate):
    return MLPHyperParameters(
        k=17,
        summarize_bert_features=True,
        learning_rate=learning_rate,
        num_epochs=30,
        hidden_layer_sizes=[2048, 256, 32],
        dropout_probability=0.1,
    )


# Searches for the best learning rate.
def compare_learning_rates():
    rates_to_compare = [1e-5, 3e-5, 5e-5, 1e-4]
    losses = [
        evaluate_with_hyperparameters(
            initial_hps_guess_with_learning_rate(learning_rate)
        )
        for learning_rate in rates_to_compare
    ]
    best_loss_idx = np.argmin(losses)
    return rates_to_compare[best_loss_idx], losses[best_loss_idx]


# compare_learning_rates()
# Best appears to be 3e-5.


# The start of hyperparameter optimzation. Reasonable guesses for everything; we'll start
# by tuning the learning rate.
def hps_guess_1_with_summarize_bert(summarize_bert):
    return MLPHyperParameters(
        k=17,
        summarize_bert_features=summarize_bert,
        learning_rate=3e-5,
        num_epochs=30,
        hidden_layer_sizes=[2048, 256, 32],
        dropout_probability=0.1,
    )


def compare_bert_summarization():
    values_to_compare = [False, True]
    losses = [
        evaluate_with_hyperparameters(hps_guess_1_with_summarize_bert(value))
        for value in values_to_compare
    ]
    best_loss_idx = np.argmin(losses)
    return values_to_compare[best_loss_idx], losses[best_loss_idx]


# Summarizing / aggregating works better


def hps_guess_2_with_hidden_layers(hidden_layers):
    return MLPHyperParameters(
        k=17,
        summarize_bert_features=True,
        learning_rate=3e-5,
        num_epochs=30,
        hidden_layer_sizes=hidden_layers,
        dropout_probability=0.1,
    )


def compare_hidden_layers():
    values_to_compare = [
        [4096, 512, 32],
    ]
    losses = [
        evaluate_with_hyperparameters(hps_guess_2_with_hidden_layers(value))
        for value in values_to_compare
    ]
    best_loss_idx = np.argmin(losses)
    return values_to_compare[best_loss_idx], losses[best_loss_idx]


# Best so far: [4096, 512, 32]


def hps_guess_3_with_dropout_probability(dropout_probability):
    return MLPHyperParameters(
        k=17,
        summarize_bert_features=True,
        learning_rate=3e-5,
        num_epochs=50,
        hidden_layer_sizes=[4096, 512, 32],
        dropout_probability=dropout_probability,
    )


def compare_dropout_probabilities():
    values_to_compare = [0.05, 0.1, 0.2, 0.3]
    losses = [
        evaluate_with_hyperparameters(hps_guess_3_with_dropout_probability(value))
        for value in values_to_compare
    ]
    best_loss_idx = np.argmin(losses)
    return values_to_compare[best_loss_idx], losses[best_loss_idx]


def hps_guess_4_with_epochs(num_epochs):
    return MLPHyperParameters(
        k=17,
        summarize_bert_features=True,
        learning_rate=3e-5,
        num_epochs=num_epochs,
        hidden_layer_sizes=[4096, 512, 32],
        dropout_probability=0.2,
    )


def compare_epochs():
    values_to_compare = [50, 100, 150, 200]
    losses = [
        evaluate_with_hyperparameters(hps_guess_4_with_epochs(value))
        for value in values_to_compare
    ]
    best_loss_idx = np.argmin(losses)
    return (
        values_to_compare[best_loss_idx],
        losses[best_loss_idx],
        values_to_compare,
        losses,
    )
