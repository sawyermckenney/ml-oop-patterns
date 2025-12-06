import torch 
import torch.nn as nn
import torch.nn.functional as F



class SimpleExample(nn.Module):
    def __init__(self):
        super(SimpleExample, self).__init__()
        self.fc1 = nn.Linear(10, 5) # Fully connected layer taking 10 input features and 5 hidden neurons 
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
model = SimpleExample()
output = model(torch.randn(1, 10))
print(output)