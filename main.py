# fabienfrfr 20220826

import torch
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
			l += [[net[c[0]][3], n[3], net[c[0]][0], n[0], n[1], i]]
			i+=1
	return np.array(l)

list_step = xli_list(Net)

### Ordering operation following position (forward)
orderForward = np.argsort(list_step[:,0])

forward_step = list_step[orderForward]

"""
ideas : facilities of real calculation of pytorch
	LOOP(index to select) # seeing with jit
"""