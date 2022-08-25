#fabienfrfr 20220825

from ctypes import *
import torch
from py import graph_gen, pRNN_net

# parameter
I,O = 64,16
BATCH = 25

graph = graph_gen.GRAPH((I,O))
net = graph.NEURON_LIST.tolist()

# Test c++ version

model = CDLL('build/pRNN') 
model.main()