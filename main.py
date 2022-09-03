# fabienfrfr 20220826

import torch, torch.nn as nn
# to improve
from py import graph_gen, pRNN_net
# baseline modif
#import prnn_baseline as prnn
# Cpp convert
import prnn_cpp

print(prnn_cpp)

### Network parameter
I,O = 16,8
BATCH = 25

### Graph construction
graph = graph_gen.GRAPH((I,O))
while len(graph.LIST_C) > 32 or len(graph.NEURON_LIST) < 5 :
	graph = graph_gen.GRAPH((I,O))
print("Generate small network for example")


List_connection = graph.LIST_C # [Position X, Layers index L, Connection index C]
Net = graph.NEURON_LIST # []

### Testing
import numpy as np 

'''
Part in reflexion / dev
'''

### Create operation forward&backward list
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

list_step = xli_list(Net)
list_step[list_step == -1] = list_step[:,3].max() + 1

### Ordering operation following position (forward)
orderForward = np.lexsort((list_step[:,-1],list_step[:,1]))

forward_step = list_step[orderForward]
prnn_located = forward_step[:,0] > forward_step[:,1]

"""
ideas : facilities of real forward calculation of pytorch
	LOOP(index to select) # seeing with jit
"""

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

Layers = nn.ModuleList(  [nn.Sequential(nn.Conv1d(p[0,1], p[0,2], 1, groups=int(p[0,1]), bias=True), nn.ReLU())] + # input
						 [nn.Sequential(nn.Linear(q[1], q[2]), nn.ReLU()) for q in p[1:-1]] +
						 [nn.Linear(p[-1,1], p[-1,2])]) # output

### Forward test (to index to select to method -> cat t addmm)
X = torch.rand(25,I, requires_grad=True)
forward = forward_step.tolist()

s = tuple(X.shape)
trace = [torch.zeros(25,q[-1], requires_grad=True) for q in p]
trace_ = [torch.zeros(25,q[-1], requires_grad=True) for q in p]
x = X.view(s[0],s[1],1)

forward_step = np.concatenate((forward_step, prnn_located[:,None]), axis=1)
group_calcul = np.split(forward_step, np.unique(forward_step[:,1], return_index=True)[1][1:])
group_calcul = [g.tolist() for g in group_calcul]

trace[0] = Layers[0](x).view(s)
trace_[0] = Layers[0](x).view(s)

for gc in group_calcul :
	tensor = []
	for step in gc :
		if step[-1] == 0 :
			tensor += [trace[step[2]][:,step[4]][:, None]]
			#grad_fn=<UnsqueezeBackward0>
		else :
			tensor += [trace[step[2]][:,step[4]][:, None].detach()]
	tensor = torch.cat(tensor, dim=1)
	trace[step[3]] = Layers[step[3]](tensor)

out = trace[-1]
print(out, out.sum().backward())
Layers.zero_grad()

# with Autograd (see https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html)

class PRNNFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, trace, layers, group_calcul):
		global attribute
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
		ctx.save_for_backward(*trace)
		return trace[-1]

	@staticmethod
	def backward(ctx, grad_output):
		result, = ctx.saved_tensors
		grad = tuple([grad_output[i]*result[i] for i in range(len(result))])
		return grad.clone()

#out_ = PRNNFunction.apply(trace, Layers, group_calcul)#.backward()
layers_p = [p for p in Layers.parameters()]

out_ = PRNNFunction.apply(trace_, Layers, group_calcul)#.backward()
print(out_) # see why he don't save backwards ! https://pytorch.org/docs/stable/notes/extending.html
