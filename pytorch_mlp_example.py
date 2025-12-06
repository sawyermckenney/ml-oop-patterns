import torch 
import torch.nn as nn
import torch.nn.functional as F

# This is a two layer Neural Network in PyTorch, what it does is it takes 10 
# input features which are the features of the data (age, gender, etc.)
# and it has 5 hidden nodes which are the hidden nodes of the Neural Network
# and it has 1 output node which is the output of the Neural Network
# the output is a sigmoid function which is a logistic function
# the sigmoid function is a function that takes a real number and squashes it to a value between 0 and 1.

class SimpleExample(nn.Module):
    def __init__(self):
        super(SimpleExample, self).__init__()
        self.fc1 = nn.Linear(10, 5) 
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
model = SimpleExample()
output = model(torch.randn(1, 10))
print(output.item())