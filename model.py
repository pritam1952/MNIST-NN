# model.py
from tensor import Tensor
from nn import Module, Linear, ReLU, Sequential, Dropout

class MNISTNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(28*28, 128)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(0.3)
        self.fc2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(0.3)
        self.fc3 = Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x