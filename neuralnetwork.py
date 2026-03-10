#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
from losses import MeanSquaredError
from optimizer import SGD


class NeuralNetwork:

    def __init__(
        self,
        epochs=100,
        batch_size=128,
        optimizer=None,
        learning_rate=0.01,
        momentum=0.90,
        verbose=False,
        loss=MeanSquaredError,
        metric=None,
        early_stopping=True,
        early_stopping_patience=10,
        random_state=42
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer if optimizer is not None else SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        self.early_stopping = early_stopping
        self.patience = early_stopping_patience
        self.random_state = random_state

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())

        if hasattr(layer, "initialize"):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]

            if y is not None:
                yield X[batch_idx], y[batch_idx]
            else:
                yield X[batch_idx], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def _save_best_weights(self):
        best_layers = []
        for layer in self.layers:
            layer_copy = copy.deepcopy(layer)
            best_layers.append(layer_copy)
        return best_layers

    def _restore_best_weights(self, best_layers):
        self.layers = best_layers

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y

        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {
            "loss": [],
            "metric": []
        }

        best_loss = np.inf
        best_layers = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            epoch_outputs = []
            epoch_targets = []

            for X_batch, y_batch in self.get_mini_batches(X, y, shuffle=True):
                # Forward
                output = self.forward_propagation(X_batch, training=True)

                # Loss gradient
                error = self.loss.derivative(y_batch, output)

                # Backward
                self.backward_propagation(error)

                epoch_outputs.append(output)
                epoch_targets.append(y_batch)

            output_all = np.concatenate(epoch_outputs, axis=0)
            y_all = np.concatenate(epoch_targets, axis=0)

            current_loss = self.loss.loss(y_all, output_all)

            if self.metric is not None:
                current_metric = self.metric(y_all, output_all)
            else:
                current_metric = None

            self.history["loss"].append(current_loss)
            self.history["metric"].append(current_metric)

            if self.verbose:
                if current_metric is not None:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {current_loss:.4f} - metric: {current_metric:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {current_loss:.4f}")

            # Early stopping
            if current_loss < best_loss - 1e-8:
                best_loss = current_loss
                best_layers = self._save_best_weights()
                wait = 0
            else:
                wait += 1

            if self.early_stopping and wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping na época {epoch}")
                break

        if self.early_stopping and best_layers is not None:
            self._restore_best_weights(best_layers)

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        raise ValueError("Sem métrica definida.")