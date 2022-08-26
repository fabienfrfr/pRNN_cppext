# fabienfrfr 20220825

import torch
from py import graph_gen, pRNN_net

# parameter
I,O = 64,16
BATCH = 25

graph = graph_gen.GRAPH((I,O))
net = graph.NEURON_LIST#.tolist()

# to trace
model = pRNN_net.pRNN(net, BATCH, I, torch.device('cpu'))

# An example input you would normally provide to your model's forward() method.
example = torch.randn(BATCH,I, requires_grad=True)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
output = traced_script_module = torch.jit.trace(model, example)

# save
traced_script_module.save("traced_enn_model.pt")

"""
The model trace change following input, jit it's not adapted for convert code to C++

But, can be used for see minimal object necessary in C++ extension !
"""