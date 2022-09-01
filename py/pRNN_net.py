#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:01:37 2021
@author: fabien
"""

import torch, torch.nn as nn
import numpy as np
#import torch.nn.functional as F

################################ Custom neural network
class pRNN(nn.Module):
	def __init__(self, NET,B,I, DEVICE, STACK=False):
		super().__init__()
		self.NET = NET # list of list
		self.BS = B
		self.STACK = STACK # mini-batch reccurence
		# list of layers
		self.Layers = nn.ModuleList( [nn.Linear(self.NET[0][2], self.NET[0][1])] +
									 [nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.NET[1:]] +
									 [nn.Sequential(nn.Conv1d(I, I, 1, groups=I, bias=True), nn.ReLU())]).to(DEVICE)
		# trace data
		self.trace = (len(NET)+1)*[None]
		# pseudo RNN (virtual input)
		self.h = [torch.zeros(B,n[1]).to(DEVICE) for n in self.NET] + [torch.zeros(B,I).to(DEVICE)]
	
	def graph2net(self, BATCH_, requires_stack = False):
		trace = (len(self.NET))*[None]
		# hidden to output (X ordered)
		for i in np.argsort(self.NET[:, 3]) :
			tensor = []
			for j,k in self.NET[i, -1] :
				# input
				if j == 0 : tensor += [self.trace[-1][BATCH_,None,k]]
				# hidden
				else :
					# pseudo-RNN (virtual input)
					if (self.NET[j, 3] >= self.NET[i, 3]) : tensor += [self.h[j][BATCH_,None,k]]
					# Non Linear input
					else :
						if requires_stack :
							tensor += [trace[j][0,None,k]]
						else :
							tensor += [trace[j][BATCH_,None,k]]
				if requires_stack :
					tensor[-1] = tensor[-1][None]
			tensor_in = torch.cat(tensor, dim=1)
			trace[i] = self.Layers[i](tensor_in)
		return i, trace
	
	def forward(self,x):
		s = tuple(x.shape)
		# Generalization of Exploitation or Training batch
		BATCH_ = np.arange(s[0])
		# input functionalization (with spread sparsing)
		self.trace[-1] = self.Layers[-1](x.view(s[0],s[1],1)).view(s)
		if self.STACK :
			trace = []
			# adapted for mini-batch but slow (python & no tensor calc avantage)
			for b in BATCH_ :
				idx_end, trace_ = self.graph2net(b, requires_stack = True)
				trace += [trace_]
				# save t+1 (and periodic bound if )
				for t in range(len(trace_)):
					if b < s[0]-1 :
						self.h[t][b+1] = trace_[t].detach()
					else :
						self.h[t][0] = trace_[t].detach()
			# reconstruct tracing
			for t in range(len(trace_)) :
				self.trace[t] = torch.cat([tt[t] for tt in trace])
		else :
			# Only adapted for SGD, if mini-batch, pseudo-rnn perturbation
			idx_end, self.trace[:-1] = self.graph2net(BATCH_)
			# save for t+1
			for t in range(len(self.trace)):
				self.h[t][BATCH_] = self.trace[t][BATCH_].detach()
		# output probs
		return self.trace[idx_end]
