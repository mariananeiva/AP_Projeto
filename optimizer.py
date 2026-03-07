#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Optimizer:
    def update(self, w, grad_loss_w):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        
        return w - self.learning_rate * self.retained_gradient

    def clone(self):
        return SGD(learning_rate=self.learning_rate, momentum=self.momentum)