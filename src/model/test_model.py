import torch
# creating custom modules with package 'nn.Module'
class LR(torch.nn.Module):
    # Object Constructor
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
    # define the forward function for prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred