#include <torch/extension.h>
#include <vector>
#include <iostream>

// pass tensor in list layer by network list graph
std::vector<torch::Tensor> prnn_graph2net(
  torch::Tensor input) {

  return {input};
}

std::vector<torch::Tensor> prnn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<torch::Tensor> prnn_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = (1 - new_cell.tanh().pow(2)) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  auto s_input_gate = torch::sigmoid(gates[0]);
  auto s_output_gate = torch::sigmoid(gates[1]);
  d_input_gate *= (1 - s_input_gate) * s_input_gate ;
  d_output_gate *= (1 - s_output_gate) * s_output_gate ;
  torch::Scalar alpha = 1.0;
  auto e = gates[2].exp();
  auto mask = (alpha * (e - 1)) < 0;
  d_candidate_cell *= (gates[2] > 0).type_as(gates[2]) + mask.type_as(gates[2]) * (alpha * e);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prnn", &prnn_forward, "PRNN forward");
  m.def("prnn", &prnn_backward, "PRNN backward");
}
