import torch
# creating custom modules with package 'nn.Module'
class quality_model_1_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(69, 1, bias = False)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        # sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        sum_tensor = self.linear(x)
        result = self.sig(sum_tensor)
        return result

class quality_model_2_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(69, 200, bias = False)
        self.linear_2 = torch.nn.Linear(200, 1, bias = False)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        # sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        sum_tensor = self.linear(x)
        sum_tensor = self.linear_2(sum_tensor)
        result = self.sig(sum_tensor)
        return result

class quality_model_3_layer(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(69, 400, bias=False)
        self.linear_2 = torch.nn.Linear(400, 200, bias = False)
        self.linear_3 = torch.nn.Linear(200, 1, bias = False)
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        # sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        sum_tensor = self.linear(x)
        sum_tensor = self.linear_2(sum_tensor)
        sum_tensor = self.linear_3(sum_tensor)
        result = self.sig(sum_tensor)
        return result