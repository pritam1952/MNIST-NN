# nn.py
from tensor import Tensor
import math

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def parameters(self):
        """Get all parameters of this module and its submodules."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def __setattr__(self, name, value):
        """Special handling for parameters and modules."""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)
    
    def train(self, mode=True):
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""
    def __init__(self, data, shape=None):
        super().__init__(data, shape, requires_grad=True)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Xavier/Glorot initialization
        k = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(self._init_weights(in_features, out_features, -k, k), (out_features, in_features))
        
        if bias:
            self.bias = Parameter(self._init_weights(out_features, 1, -k, k), (out_features,))
        else:
            self.bias = None
    
    def _init_weights(self, rows, cols, a, b):
        """Initialize weights uniformly between a and b."""
        if cols == 1:  # For bias
            return [a + (b - a) * random.random() for _ in range(rows)]
        else:  # For weight matrix
            return [[a + (b - a) * random.random() for _ in range(cols)] for _ in range(rows)]
    
    def forward(self, x):
        output = x @ self.weight.transpose()
        if self.bias is not None:
            # Reshape bias for broadcasting
            bias_shaped = self.bias.reshape(1, self.out_features)
            output = output + bias_shaped
        return output

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.__setattr__(f"layer{i}", layer)
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Softmax(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim  # Dimension along which to apply softmax
    
    def forward(self, x):
        # For now, only supporting softmax along dim=1 (for 2D tensors)
        if self.dim != 1 or len(x.shape) != 2:
            raise NotImplementedError("Softmax only implemented for 2D tensors along dim=1")
        
        # Subtract max for numerical stability
        max_vals = []
        for i in range(x.shape[0]):
            row_max = max(x.data[i])
            max_vals.append(row_max)
        
        # Compute exponentials
        exps = []
        for i in range(x.shape[0]):
            row = []
            for j in range(x.shape[1]):
                row.append(math.exp(x.data[i][j] - max_vals[i]))
            exps.append(row)
        
        # Normalize by sum
        result_data = []
        for i in range(x.shape[0]):
            row_sum = sum(exps[i])
            row = [exp_val / row_sum for exp_val in exps[i]]
            result_data.append(row)
        
        result = Tensor(result_data, x.shape, x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor.ones(*x.shape)
                
                # Compute Jacobian-vector product directly
                # For each sample in the batch
                grad_x_data = []
                for i in range(x.shape[0]):
                    row_grad = []
                    for j in range(x.shape[1]):
                        # For each output dimension
                        dx_j = 0
                        for k in range(x.shape[1]):
                            # Kronecker delta - 1 if j==k, 0 otherwise
                            delta_jk = 1.0 if j == k else 0.0
                            # S_j * (delta_jk - S_k)
                            dx_j += result_data[i][j] * (delta_jk - result_data[i][k]) * grad.data[i][k]
                        row_grad.append(dx_j)
                    grad_x_data.append(row_grad)
                
                x.backward(Tensor(grad_x_data, x.shape))
            
            result._grad_fn = _backward
            result.children = [x]
        
        return result

class Flatten(Module):
    def forward(self, x):
        batch_size = x.shape[0]
        flattened_size = math.prod(x.shape) // batch_size
        return x.reshape(batch_size, flattened_size)

import random

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Create dropout mask
        mask_data = []
        for i in range(x.shape[0]):
            if len(x.shape) == 1:
                mask_data.append(1.0 if random.random() > self.p else 0.0)
            else:
                row_mask = []
                for j in range(x.shape[1]):
                    row_mask.append(1.0 if random.random() > self.p else 0.0)
                mask_data.append(row_mask)
        
        self.mask = Tensor(mask_data, x.shape)
        
        # Scale by 1/(1-p) to maintain expectation
        scale = 1.0 / (1.0 - self.p)
        return x * self.mask * scale