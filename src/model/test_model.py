import torch
import util

class quality_model_1_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self, base_context_count):
        super().__init__()
        self.tensor_length = pow(5, base_context_count) + 8
        self.linear = torch.nn.Linear(self.tensor_length, 1, bias = True)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        sum_tensor = self.linear(x)
        result = self.sig(sum_tensor)
        return result
