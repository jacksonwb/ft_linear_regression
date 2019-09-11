#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  linreg.py                                                                   #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Wednesday Sep 2019 2:30:08 pm                                     #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import csv

def step_gradient(X, Y, params, learning_rate):
	Y_hat = params[0] * X + params[1]
	grad_0 = -2 * sum(X * (Y - Y_hat)) / len(X)
	grad_1 = -2 * sum(Y - Y_hat) / len(X)
	return params - (np.array([grad_0, grad_1]) * learning_rate)

def normalize(X):
	return (X - np.min(X)) / (np.max(X) - np.min(X))

def denormalize(X, width, min_val):
	return X * width + min_val

def normalize_point(point, width, min_val):
	return (point - min_val) / (width)

def denormalize_point(point, width, min_val):
	return point * width + min_val

def solve_point	(x, params, width, min_val):
	x_norm = normalize_point(x, width, min_val)
	return x_norm * params[0] + params[1]

class LinReg:
	def __init__(self):
		self.solved = False
	def train(self, X, Y, learning_rate, max_iter, plot=False):
		params = [0, 0]
		X_norm = normalize(X)
		for i in range(max_iter):
			params = step_gradient(X_norm, Y, params, learning_rate)
		self.params = params.tolist()
		self.min_val = np.min(X)
		self.width = np.max(X) - self.min_val
		self.solved = True
		if plot:
			plt.scatter(x=X, y=Y)
			plt.plot(X, X_norm * params[0] + params[1])
			plt.show()
	def save_model(self, model_name='lin_reg_model.csv'):
		if not self.solved:
			print("Model has not been solved")
		else:
			with open(model_name, mode='w') as model_file:
				model_writer = csv.writer(model_file)
				model_writer.writerow(self.params + [self.width, self.min_val])
	def load_model(self, model_name='lin_reg_model.csv'):
		try:
			with open(model_name) as model_file:
				model_reader = csv.reader(model_file)
				model = next(model_reader)
		except Exception as e:
			print(e)
			return
		if not model or len(model) != 4:
			print('invalid model')
		else:
			self.params = list(map(float, model[0:2]))
			self.width = float(model[2])
			self.min_val = float(model[3])
			self.solved = True
	def solve_for(self, x):
		if not self.solved:
			print('Model has not been solved')
		else:
			x_norm = normalize_point(x, self.width, self.min_val)
			return x_norm * self.params[0] + self.params[1]