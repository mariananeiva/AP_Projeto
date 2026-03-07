#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from layers import DenseLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from metrics import mse

class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = MeanSquaredError,
                 metric:callable = mse,
                 early_stopping_patience = 10): # Adicionada paciência para Early Stopping
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.patience = early_stopping_patience

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None, shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        # Ciclo invertido: da última camada para a primeira
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        best_loss = np.inf
        wait = 0

        for epoch in range(1, self.epochs + 1):
            output_x_ = []
            y_ = []
            
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # 1. Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # 2. Calcular o erro (gradiente) da perda
                error = self.loss.derivative(y_batch, output)
                # 3. Backward propagation (Treino)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            # Cálculo de métricas da época
            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)
            current_loss = self.loss.loss(y_all, output_x_all)
            
            # Early Stopping
            if current_loss < best_loss:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1
            
            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            self.history[epoch] = {'loss': current_loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {current_loss:.4f} - {metric_s}")

            # Parar se o erro não melhorar durante 'patience' épocas
            if wait >= self.patience:
                if self.verbose:
                    print(f"Early stopping na época {epoch}")
                break

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("Sem métrica definida.")