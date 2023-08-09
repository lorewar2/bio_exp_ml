import torch
# creating custom modules with package 'nn.Module'
class quality_model(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(69, 1, bias=False)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        # sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        sum_tensor = self.linear(x)
        result = self.sig(sum_tensor)
        return result