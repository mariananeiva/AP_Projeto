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
        self.velocity = None

    def update(self, w, grad_loss_w):
        if self.velocity is None:
            self.velocity = np.zeros_like(w)

        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.velocity

    def clone(self):
        return SGD(learning_rate=self.learning_rate, momentum=self.momentum)