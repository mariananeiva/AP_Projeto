#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer


class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        clipped_input = np.clip(input, -500, 500)
        return 1 / (1 + np.exp(-clipped_input))

    def derivative(self, input):
        sig = self.activation_function(input)
        return sig * (1 - sig)


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return (input > 0).astype(float)


class SoftmaxActivation(ActivationLayer):
    """
    Ativação para classificação multi-classe.
    Converte os scores da última camada em probabilidades por classe.
    """

    def activation_function(self, input):
        shifted = input - np.max(input, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, input):
        """
        A derivada completa do softmax é jacobiana. Nesta implementação
        simplificada, esta camada deve ser usada em conjunto com
        CategoricalCrossEntropy quando o gradiente já é tratado de forma
        combinada no cálculo da loss.
        """
        return np.ones_like(input)