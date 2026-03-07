#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MeanAbsoluteError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1) / y_true.size


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


## se quisermos só testar entre duas categorias tipo AI ou Humano
class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / p - (1 - y_true) / (1 - p)) / y_true.size


## se quisermos testar todas as categorias diferentes que vamos ter 
class CategoricalCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p)) / y_true.shape[0]

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0] 
    

