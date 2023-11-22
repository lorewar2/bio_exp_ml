import torch
import util

class quality_model_1_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self, base_context_count, extra_count):
        super().__init__()
        self.tensor_length = pow(5, base_context_count) + extra_count
        self.linear = torch.nn.Linear(self.tensor_length, 1, bias = False)
        #self.linear2 = torch.nn.Linear(self.tensor_length * 2, self.tensor_length, bias = True)
        #self.linear3 = torch.nn.Linear(self.tensor_length, 1, bias = True)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        x = self.linear(x)
        x = self.sig(x)
        #out = self.linear2(out)
        #out = self.linear3(out)
        return x