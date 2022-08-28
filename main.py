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

"""
ideas : facilities of real forward calculation of pytorch
	LOOP(index to select) # seeing with jit
"""

### Layers
Layers = np.zeros(())
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
X = torch.rand(25,I)
forward = forward_step.tolist()

s = tuple(X.shape)

x = X.view(s[0],s[1],1)

x = Layers[0](x).view(s)

trace = []
for step in forward :
	trace  += [(step[2],step[-2])]
print(trace)