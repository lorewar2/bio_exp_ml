import torch
import util

# creating custom modules with package 'nn.Module'
class quality_model(torch.nn.Module):
    # Object Constructor
    def __init__(self, base_context, layer_count):
        super().__init__()
        self.layer_count = layer_count
        # make a layer vector
        layer_node_count = []
        # calculate the one hot encoding count
        one_hot_count = pow(4, base_context)
        # calibrate first layer
        layer_node_count.append(one_hot_count + 5)
        # calibrate the rest of layers
        for index in range(0, layer_count - 1):
            layer_node_count.append(200)
        # calibrate the last layer 1 node
        layer_node_count.append(1)
        self.layer_list = []
        # make the layers
        for index in range(0, layer_count):
            self.layer_list.append(torch.nn.Linear(layer_node_count[index], layer_node_count[index + 1], bias = False))
        self.sig = torch.nn.Sigmoid()
    # define the forward function for prediction
    def forward(self, x):
        sum_tensor = x
        for index in range(0, self.layer_count):
            sum_tensor = self.layer_list[index](sum_tensor)
        result = self.sig(sum_tensor)
        return result

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