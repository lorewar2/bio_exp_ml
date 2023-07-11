import torch
# creating custom modules with package 'nn.Module'
class quality_model(torch.nn.Module):
    # Object Constructor
    def __init__(self):
        super().__init__()
        self.linear_context = torch.nn.Linear(3, 1)
        self.linear_parallel = torch.nn.Linear(4, 1)
        self.linear_quality = torch.nn.Linear(1, 1)
        self.linear_sum = torch.nn.Linear(3, 1)
    # define the forward function for prediction
    def forward(self, x):
        context_tensor = x[:, :3]
        quality_tensor = x[:, 3:4]
        parallel_tensor = x[:, 4:]
        context_tensor = self.linear_context(context_tensor)
        parallel_tensor = self.linear_parallel(parallel_tensor)
        quality_tensor = self.linear_quality(quality_tensor)
        sum_tensor = torch.concat((context_tensor, parallel_tensor, quality_tensor), 1)
        result = self.linear_sum(sum_tensor)
        return result