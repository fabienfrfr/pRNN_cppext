#fabienfrfr 20220826
import numpy as np
import torch, torch.nn as nn
from torch.autograd import Function

"""
ONLY IN PYTHON HERE, IT'S A pRNN Alternative !!
Convert in C Extension after !
Limitation : use only list and basic torch (included in C header), numpy it's forbidden ! Complex torch function it's also forbidden, see minimal utilization in traced jit model.
"""

### Operation forward&backward constructor
'''
i -> initial, f-> final, j = index
X = Position, L = Layers, C/O = connector/output of layers

'''
def xli_list(net) :
	l = []
	for n in net :
		i = 0 
		for c in n[-1]:
			# Xi, Xf, Li, Lf, Cj, Oj
			if c[0] == 0 :
				l += [[0, n[3], 0, n[0], c[1], i]]
			else :
				l += [[net[c[0]][3], n[3], net[c[0]][0], n[0], c[1], i]]
			i+=1
	return np.array(l)

### Model baseline
class PRNNFunction(Function):
	@staticmethod
	def forward(ctx, trace, layers, group_calcul):
		#ctx.save_for_backward(*trace)
		for gc in group_calcul :
			tensor = []
			for step in gc :
				if step[-1] == 0 :
					tensor += [trace[step[2]].select(1,step[4]).unsqueeze(1)]
					#requires_grad = True (only, no grad_fn=<UnsqueezeBackward0>...)
					#torch.autograd.grad(input_, tensor)
				else :
					tensor += [trace[step[2]].select(1,step[4]).unsqueeze(1).detach()]
					tensor[-1].requires_grad = True
			tensor = torch.cat(tensor, dim=1)
			attribute = getattr(layers, str(step[3]))
			if isinstance(layers[step[3]], torch.nn.Sequential):
				attribute = getattr(attribute, "0")
			weight = torch.t(attribute.weight)
			trace[step[3]] = torch.addmm(attribute.bias, tensor, weight, beta=1, alpha=1)
		#trace = [t.clone() for t in trace]
		#trace = prnn_cpp.forward(trace, layers, group_calcul)
		ctx.save_for_backward(*trace)
		return trace[-1]

	@staticmethod
	def backward(ctx, grad_output):
		result, = ctx.saved_tensors
		'''
		here, construct all backward operation
		it's here to find "grad_fn" part, and that explain why you don't have "grad_fn" in forward staticmethod !
		NEED REFLEXION HERE
		'''
		grad = tuple([grad_output[i]*result[i] for i in range(len(result))])
		#grad = prnn_cpp.backward(tuple([grad_output[i]*result[i] for i in range(len(result))]))
		return grad


class pRNN(nn.Module):
	def __init__(self, NET, BATCH=25):
		super(pRNN, self).__init__()
		self.net = graph.NEURON_LIST
		self.batch = BATCH
		# Create operation forward&backward list
		list_step = xli_list(self.net)
		list_step[list_step == -1] = list_step[:,3].max() + 1
		### Ordering operation following position (forward)
		orderForward = np.lexsort((list_step[:,-1],list_step[:,1]))

		forward_step = list_step[orderForward]
		prnn_located = forward_step[:,0] > forward_step[:,1]
		### Layers
		real_out = np.unique(list_step[:,[2,-2]], axis=0)
		out_layers = np.unique(real_out[:,0], return_counts=True)
		in_layers = np.unique(list_step[:,3], return_counts=True)

		N = len(in_layers[0])+1
		p = np.zeros((N,3)).astype(int)

		p[:,0] = np.arange(N).astype(int)
		p[(0,-1),(1,-1)] = I, O
		p[in_layers[0],1] = in_layers[1]
		p[out_layers[0],2] = out_layers[1]

		self.Layers = nn.ModuleList(  [nn.Sequential(nn.Conv1d(p[0,1], p[0,2], 1, groups=int(p[0,1]), bias=True), nn.ReLU())] + # input
								 [nn.Sequential(nn.Linear(q[1], q[2]), nn.ReLU()) for q in p[1:-1]] +
								 [nn.Linear(p[-1,1], p[-1,2])]) # output
		### forward constructor
		forward_step = np.concatenate((forward_step, prnn_located[:,None]), axis=1)
		group_calcul = np.split(forward_step, np.unique(forward_step[:,1], return_index=True)[1][1:])
		self.group_calcul = [g.tolist() for g in group_calcul]
		self.trace = [torch.zeros(self.batch,q[-1], requires_grad=True) for q in p]

	def forward(self, x):
		# input
		s = tuple(x.shape)
		self.trace[0] = self.Layers[0](x.view(s[0],s[1],1)).view(s)
		# graph_net
		return PRNNFunction.apply(self.trace, self.Layers, self.group_calcul)

### Testing
if __name__ == '__main__':
	# package for test
	from py import graph_gen
	# parameter
	I,O = 16,8
	BATCH = 25
	### Graph construction
	graph = graph_gen.GRAPH((I,O))
	while len(graph.LIST_C) > 32 or len(graph.NEURON_LIST) < 5 :
		graph = graph_gen.GRAPH((I,O))
	print("Generate small network for example")
	Net = graph.NEURON_LIST # []
	# input 
	X = torch.rand(BATCH,I, requires_grad=True)
	# model
	model = pRNN(Net, BATCH)
	# out
	out = model(X)
	print(out)