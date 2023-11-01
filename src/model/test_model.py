import torch
import util

class quality_model_1_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self, base_context_count):
        super().__init__()
        self.tensor_length = pow(5, base_context_count) + 8
        self.linear = torch.nn.Linear(self.tensor_length - 5, 1, bias = True)
        self.linear2 = torch.nn.Linear(5, 1, bias = True)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        x1 = self.linear(x[0:self.tensor_length - 5])
        x2 = self.linear2(x[self.tensor_length - 5:])
        result = self.sig(x1 + x2)
        return result