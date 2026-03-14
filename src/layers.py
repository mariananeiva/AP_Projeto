#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer(Layer):
    
    def __init__(self, n_units, input_shape=None, l2_lambda=0.0, initialization="xavier"):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.l2_lambda = l2_lambda
        self.initialization = initialization

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self.dweights = None
        self.dbiases = None
        
    def initialize(self, optimizer):
        input_dim = self.input_shape()[0]

        if self.initialization == "he":
            std = np.sqrt(2.0 / input_dim)
            self.weights = np.random.randn(input_dim, self.n_units) * std

        elif self.initialization == "xavier":
            limit = np.sqrt(6.0 / (input_dim + self.n_units))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.n_units))

        else:
            self.weights = np.random.randn(input_dim, self.n_units) * 0.01

        self.biases = np.zeros((1, self.n_units))
        
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)

        self.dweights = np.dot(self.input.T, output_error)
        self.dbiases = np.sum(output_error, axis=0, keepdims=True)

        if self.l2_lambda > 0:
            self.dweights += self.l2_lambda * self.weights
    
        self.weights = self.w_opt.update(self.weights, self.dweights)
        self.biases = self.b_opt.update(self.biases, self.dbiases)

        return input_error
 
    def output_shape(self):
        return (self.n_units,)
 
    def output_shape(self):
         return (self.n_units,) 

# evitar o overfitting
class DropoutLayer(Layer):
    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward_propagation(self, input_data, training):
        if training:
            keep_prob = 1 - self.probability
            self.mask = np.random.binomial(1, keep_prob, size=input_data.shape) / keep_prob
            return input_data * self.mask

        self.mask = None
        return input_data

    def backward_propagation(self, output_error):
        if self.mask is None:
            return output_error
        return output_error * self.mask

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0