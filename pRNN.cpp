#include <torch/torch.h>
#include <iostream>


int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}

/*

struct pRNN : torch::nn::Module {

  // init
  pRNN(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }

  // forward
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }

  // scope operator of variable declaration
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};


int main() {
  pRNN net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
}


class Net(torch.nn.Module):
  def __init__(self, N, M):
      super(Net, self).__init__()
      # Registered as a submodule behind the scenes
      self.linear = torch.nn.Linear(N, M)
      self.another_bias = torch.nn.Parameter(torch.rand(M))

  def forward(self, input):
    return self.linear(input) + self.another_bias
*/