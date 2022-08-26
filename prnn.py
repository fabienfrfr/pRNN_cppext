#fabienfrfr 20220826
import math,time
import torch, torch.nn as nn
from torch.autograd import Function

import prnn_cpp

class PRNNFunction(Function):
	@staticmethod
	def forward(ctx, input, weights, bias, old_h, old_cell):
		outputs = prnn_cpp.forward(input, weights, bias, old_h, old_cell)
		new_h, new_cell = outputs[:2]
		variables = outputs[1:] + [weights]
		ctx.save_for_backward(*variables)

		return new_h, new_cell

	@staticmethod
	def backward(ctx, grad_h, grad_cell):
		d_old_h, d_input, d_weights, d_bias, d_old_cell = prnn_cpp.backward(
			grad_h, grad_cell, *ctx.saved_variables)
		return d_input, d_weights, d_bias, d_old_h, d_old_cell


class pRNN(nn.Module):
	def __init__(self, input_features, state_size):
		super(pRNN, self).__init__()
		self.input_features = input_features
		self.state_size = state_size
		self.weights = nn.Parameter(
			torch.Tensor(3 * state_size, input_features + state_size))
		self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.state_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, +stdv)

	def forward(self, input, state):
		return PRNNFunction.apply(input, self.weights, self.bias, *state)

if __name__ == '__main__':
	# parameter
	batch_size = 16
	input_features = 32
	state_size = 128
	# import
	model = pRNN(input_features, state_size)
	# input data
	X = torch.randn(batch_size, input_features)
	h = torch.randn(batch_size, state_size)
	C = torch.randn(batch_size, state_size)
	# output model
	start = time.time()
	new_h, new_C = model(X, (h, C))
	(new_h.sum() + new_C.sum()).backward()
	print('Forward&Backward: {:.3f} s'.format(time.time() - start))
	print(new_h.shape, new_C.shape)