# fabienfrfr 20220825

import torch
from py import graph_gen, pRNN_net

# parameter
I,O = 16,8
BATCH = 25

graph = graph_gen.GRAPH((I,O))
while len(graph.LIST_C) > 32 or len(graph.NEURON_LIST) < 5 :
	graph = graph_gen.GRAPH((I,O))
print("Generate small network for example")

net = graph.NEURON_LIST#.tolist()

# to trace
model = pRNN_net.pRNN(net, BATCH, I, torch.device('cpu'))
model_stacked = pRNN_net.pRNN(net, BATCH, I, torch.device('cpu'), STACK=True)

# An example input you would normally provide to your model's forward() method.
example = torch.randn(BATCH,I, requires_grad=True)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module_stacked = torch.jit.trace(model_stacked, example)

# save
traced_script_module.save("traced_enn_model.pt")
traced_script_module_stacked.save("traced_enn_model_stacked.pt")

"""
The model trace change following input, jit it's not adapted for convert code to C++

But, can be used for see minimal object necessary in C++ extension !
"""