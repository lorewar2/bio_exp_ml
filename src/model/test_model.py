import torch
# creating custom modules with package 'nn.Module'
class quality_model(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear_context = torch.nn.Linear(3, 10)
        self.linear_parallel = torch.nn.Linear(4, 10)
        self.linear_quality = torch.nn.Linear(1, 10)
        self.linear_sum = torch.nn.Linear(30, 1)
        self.activation = torch.nn.ReLU()
    # define the forward function for prediction
    def forward(self, x):
        context_tensor = x[:, :3]
        quality_tensor = x[:, 3:4]
        parallel_tensor = x[:, 4:]
        context_tensor = self.linear_context(context_tensor)
        context_tensor = self.activation(context_tensor)
        parallel_tensor = self.linear_parallel(parallel_tensor)
        parallel_tensor = self.activation(parallel_tensor)
        quality_tensor = self.linear_quality(quality_tensor)
        quality_tensor = self.activation(quality_tensor)
        sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        result = self.linear_sum(sum_tensor)
        return result