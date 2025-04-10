# optim.py
from tensor import Tensor

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters
    
    def zero_grad(self):
        """Reset gradients of all parameters."""
        for param in self.parameters:
            param.grad = None
    
    def step(self):
        """Update parameters based on gradients."""
        raise NotImplementedError("Optimizer.step() not implemented")

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
        # Initialize velocity for each parameter
        for i, param in enumerate(self.parameters):
            self.velocity[i] = Tensor.zeros(*param.shape)
    
    def step(self):
        """Update parameters using stochastic gradient descent."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Extract grad data
            grad_data = param.grad.data
            
            # Add weight decay if specified
            if self.weight_decay > 0:
                flat_param = param.flatten().data
                flat_grad = param.grad.flatten().data
                
                for j in range(len(flat_grad)):
                    flat_grad[j] += self.weight_decay * flat_param[j]
                
                grad_data = param.grad.data
            
            # Apply momentum
            velocity_data = self._element_wise_op(
                self.velocity[i].data,
                grad_data,
                lambda v, g: self.momentum * v - self.lr * g
            )
            
            # Update velocity
            self.velocity[i] = Tensor(velocity_data, param.shape)
            
            # Update parameter
            param_data = self._element_wise_op(
                param.data,
                velocity_data,
                lambda p, v: p + v
            )
            
            # Replace parameter data
            param.data = param_data
    
    def _element_wise_op(self, data1, data2, op):
        """Apply an element-wise operation recursively."""
        if not isinstance(data1, (list, tuple)):
            if not isinstance(data2, (list, tuple)):
                return op(data1, data2)
        
        return [self._element_wise_op(x, y, op) for x, y in zip(data1, data2)]