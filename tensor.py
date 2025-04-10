# tensor.py
import math
import random
from typing import List, Tuple, Union, Callable, Optional

class Tensor:
    def __init__(self, data=None, shape=None, requires_grad=False):
        self.data = data if data is not None else []
        self.shape = shape if shape is not None else self._infer_shape(self.data)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.children = []
        
    def _infer_shape(self, data):
        """Recursively determine the shape of the tensor from the data."""
        if not isinstance(data, (list, tuple)):
            return ()
        if not data:
            return (0,)
        return (len(data),) + self._infer_shape(data[0])
    
    def reshape(self, *shape):
        """Reshape the tensor to the given shape."""
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to {shape}")
        
        flat_data = self.flatten().data
        new_tensor = Tensor(self._reshape_data(flat_data, shape), shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                self.backward(grad.reshape(self.shape))
            new_tensor._grad_fn = _backward
            new_tensor.children = [self]
            
        return new_tensor
    
    def _reshape_data(self, flat_data, shape):
        """Reshape flat data into the given shape."""
        if len(shape) == 1:
            return flat_data
        
        result = []
        items_per_sublist = math.prod(shape[1:])
        for i in range(0, len(flat_data), items_per_sublist):
            sublist = flat_data[i:i+items_per_sublist]
            result.append(self._reshape_data(sublist, shape[1:]))
        return result
    
    def flatten(self):
        """Flatten the tensor to a 1D array."""
        flat_data = self._flatten_data(self.data)
        new_shape = (len(flat_data),)
        new_tensor = Tensor(flat_data, new_shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                self.backward(grad.reshape(self.shape))
            new_tensor._grad_fn = _backward
            new_tensor.children = [self]
            
        return new_tensor
    
    def _flatten_data(self, data):
        """Recursively flatten the data."""
        if not isinstance(data, (list, tuple)):
            return [data]
        result = []
        for item in data:
            result.extend(self._flatten_data(item))
        return result
    
    def backward(self, grad=None):
        """Compute gradients with backpropagation."""
        if grad is None:
            if len(self.shape) > 0 and math.prod(self.shape) != 1:
                raise ValueError("Cannot compute backward pass on non-scalar tensor without gradient")
            grad = Tensor([1.0]) if not self.shape else Tensor(1.0)
        
        if self.grad is None:
            if len(grad.shape) == 0:  # Handle scalar grad
                self.grad = Tensor([grad.data], (1,))
            else:
                self.grad = Tensor(grad.data, grad.shape)
        else:
            # Add the incoming gradient to existing gradient
            self.grad = self.grad + grad
        
        if self._grad_fn:
            self._grad_fn(grad)
    
    @classmethod
    def zeros(cls, *shape, requires_grad=False):
        """Create a tensor filled with zeros."""
        if len(shape) == 0:
            return cls(0.0, (), requires_grad)
        
        def create_nested_zeros(shape):
            if len(shape) == 1:
                return [0.0] * shape[0]
            return [create_nested_zeros(shape[1:]) for _ in range(shape[0])]
        
        data = create_nested_zeros(shape)
        return cls(data, shape, requires_grad)
    
    @classmethod
    def ones(cls, *shape, requires_grad=False):
        """Create a tensor filled with ones."""
        if len(shape) == 0:
            return cls(1.0, (), requires_grad)
        
        def create_nested_ones(shape):
            if len(shape) == 1:
                return [1.0] * shape[0]
            return [create_nested_ones(shape[1:]) for _ in range(shape[0])]
        
        data = create_nested_ones(shape)
        return cls(data, shape, requires_grad)
    
    @classmethod
    def rand(cls, *shape, requires_grad=False):
        """Create a tensor filled with random values."""
        if len(shape) == 0:
            return cls(random.random(), (), requires_grad)
        
        def create_nested_rand(shape):
            if len(shape) == 1:
                return [random.random() for _ in range(shape[0])]
            return [create_nested_rand(shape[1:]) for _ in range(shape[0])]
        
        data = create_nested_rand(shape)
        return cls(data, shape, requires_grad)
    
    @classmethod
    def randn(cls, *shape, requires_grad=False):
        """Create a tensor filled with random values from a standard normal distribution."""
        if len(shape) == 0:
            # Box-Muller transform for Gaussian random numbers
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return cls(z, (), requires_grad)
        
        def create_nested_randn(shape):
            if len(shape) == 1:
                return [cls.randn(requires_grad=False).data for _ in range(shape[0])]
            return [create_nested_randn(shape[1:]) for _ in range(shape[0])]
        
        data = create_nested_randn(shape)
        return cls(data, shape, requires_grad)
    
    def __getitem__(self, idx):
        """Get an item or slice from the tensor."""
        if isinstance(idx, int):
            if idx >= len(self.data) or idx < -len(self.data):
                raise IndexError(f"Index {idx} out of range for tensor of shape {self.shape}")
            
            item_data = self.data[idx]
            new_shape = self.shape[1:] if len(self.shape) > 1 else ()
            new_tensor = Tensor(item_data, new_shape, self.requires_grad)
            
            if self.requires_grad:
                def _backward(grad=None):
                    if grad is None:
                        grad = Tensor([1.0])
                    
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    
                    grad_data = self.grad.data
                    for i in range(len(self.data)):
                        if i == idx:
                            if isinstance(grad_data[i], list):
                                for j in range(len(grad_data[i])):
                                    if j < len(grad.data):
                                        grad_data[i][j] += grad.data[j]
                            else:
                                grad_data[i] += grad.data
                
                new_tensor._grad_fn = _backward
                new_tensor.children = [self]
            
            return new_tensor
        
        # Handle slice - simplified for clarity
        raise NotImplementedError("Slicing not implemented yet")
    
    def __add__(self, other):
        """Add two tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Ensure shapes are compatible
        if self.shape != other.shape:
            raise ValueError(f"Cannot add tensors of shapes {self.shape} and {other.shape}")
        
        # Perform addition
        result_data = self._element_wise_op(self.data, other.data, lambda x, y: x + y)
        result = Tensor(result_data, self.shape, self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                if self.requires_grad:
                    self.backward(grad)
                
                if other.requires_grad:
                    other.backward(grad)
            
            result._grad_fn = _backward
            result.children = [self, other]
        
        return result
    
    def __mul__(self, other):
        """Multiply two tensors element-wise."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Ensure shapes are compatible for element-wise multiplication
        if self.shape != other.shape:
            raise ValueError(f"Cannot multiply tensors of shapes {self.shape} and {other.shape}")
        
        # Perform multiplication
        result_data = self._element_wise_op(self.data, other.data, lambda x, y: x * y)
        result = Tensor(result_data, self.shape, self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                if self.requires_grad:
                    self.backward(grad * other)
                
                if other.requires_grad:
                    other.backward(grad * self)
            
            result._grad_fn = _backward
            result.children = [self, other]
        
        return result
    
    def __neg__(self):
        """Negate the tensor."""
        result_data = self._element_wise_op(self.data, None, lambda x, _: -x)
        result = Tensor(result_data, self.shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                self.backward(-grad)
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def __sub__(self, other):
        """Subtract other from self."""
        return self + (-other)
    
    def __rsub__(self, other):
        """Subtract self from other."""
        return (-self) + other
    
    def __truediv__(self, other):
        """Divide self by other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Perform division
        result_data = self._element_wise_op(self.data, other.data, lambda x, y: x / y)
        result = Tensor(result_data, self.shape, self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                if self.requires_grad:
                    self.backward(grad / other)
                
                if other.requires_grad:
                    other.backward(-grad * self / (other * other))
            
            result._grad_fn = _backward
            result.children = [self, other]
        
        return result
    
    def __rtruediv__(self, other):
        """Divide other by self."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def _element_wise_op(self, data1, data2, op):
        """Apply an element-wise operation recursively."""
        if not isinstance(data1, (list, tuple)):
            if data2 is None:
                return op(data1, None)
            if not isinstance(data2, (list, tuple)):
                return op(data1, data2)
        
        if data2 is None:
            return [self._element_wise_op(x, None, op) for x in data1]
        
        return [self._element_wise_op(x, y, op) for x, y in zip(data1, data2)]
    
    def sum(self, dim=None):
        """Sum all elements or along a specific dimension."""
        if dim is None:
            # Sum all elements
            flat_data = self.flatten().data
            result = sum(flat_data)
            result_tensor = Tensor(result, (), self.requires_grad)
            
            if self.requires_grad:
                def _backward(grad=None):
                    if grad is None:
                        grad = Tensor([1.0])
                    
                    # Gradient is 1 for each element
                    ones = Tensor.ones(*self.shape)
                    if isinstance(grad.data, (list, tuple)):
                        grad_value = grad.data[0]
                    else:
                        grad_value = grad.data
                    self.backward(ones * grad_value)
                
                result_tensor._grad_fn = _backward
                result_tensor.children = [self]
            
            return result_tensor
        
        # Sum along specific dimension
        raise NotImplementedError("Sum along specific dimension not implemented yet")
    
    def mean(self):
        """Calculate the mean of all elements."""
        flat_tensor = self.flatten()
        sum_tensor = flat_tensor.sum()
        count = len(flat_tensor.data)
        result = sum_tensor / count
        
        return result
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("Both tensors must have at least 2 dimensions for matmul")
        
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}")
        
        # For simplicity, handle only 2D matrix multiplication
        if len(self.shape) > 2 or len(other.shape) > 2:
            raise NotImplementedError("Batched matrix multiplication not implemented")
        
        # Do matrix multiplication
        result_shape = (self.shape[0], other.shape[1])
        result_data = []
        
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                dot_product = 0.0
                for k in range(self.shape[1]):
                    dot_product += self.data[i][k] * other.data[k][j]
                row.append(dot_product)
            result_data.append(row)
        
        result = Tensor(result_data, result_shape, self.requires_grad or other.requires_grad)
        
        if self.requires_grad or other.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor.ones(*result_shape)
                
                if self.requires_grad:
                    # dL/dA = dL/dC @ B.T
                    grad_self = grad @ self._transpose(other.data)
                    self.backward(Tensor(grad_self, (self.shape[0], self.shape[1])))
                
                if other.requires_grad:
                    # dL/dB = A.T @ dL/dC
                    grad_other = self._transpose(self.data) @ grad
                    other.backward(Tensor(grad_other, (other.shape[0], other.shape[1])))
            
            result._grad_fn = _backward
            result.children = [self, other]
        
        return result
    
    def _transpose(self, data):
        """Transpose 2D data."""
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        transposed = [[data[j][i] for j in range(rows)] for i in range(cols)]
        return transposed
    
    def transpose(self):
        """Transpose the tensor."""
        if len(self.shape) == 0:
            return self
        
        if len(self.shape) == 1:
            # Convert 1D tensor to 2D column vector then transpose
            reshaped = self.reshape(self.shape[0], 1)
            transposed_data = self._transpose(reshaped.data)
            new_shape = (1, self.shape[0])
        elif len(self.shape) == 2:
            transposed_data = self._transpose(self.data)
            new_shape = (self.shape[1], self.shape[0])
        else:
            raise NotImplementedError("Transpose for tensors with more than 2 dimensions not implemented")
        
        result = Tensor(transposed_data, new_shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor.ones(*new_shape)
                self.backward(grad.transpose())
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def exp(self):
        """Exponential function applied element-wise."""
        result_data = self._element_wise_op(self.data, None, lambda x, _: math.exp(x))
        result = Tensor(result_data, self.shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                self.backward(grad * result)
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def log(self):
        """Natural logarithm applied element-wise."""
        result_data = self._element_wise_op(self.data, None, lambda x, _: math.log(x))
        result = Tensor(result_data, self.shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                self.backward(grad / self)
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def __pow__(self, power):
        """Power function applied element-wise."""
        power_tensor = power if isinstance(power, Tensor) else Tensor(power)
        
        result_data = self._element_wise_op(
            self.data, 
            power_tensor.data if power_tensor.shape else power_tensor.data, 
            lambda x, y: x ** y
        )
        
        result = Tensor(result_data, self.shape, self.requires_grad or power_tensor.requires_grad)
        
        if self.requires_grad or power_tensor.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                if self.requires_grad:
                    # d(x^n)/dx = n * x^(n-1)
                    self.backward(grad * power_tensor * self ** (power_tensor - 1))
                
                if power_tensor.requires_grad:
                    # d(x^n)/dn = x^n * ln(x)
                    power_tensor.backward(grad * result * self.log())
            
            result._grad_fn = _backward
            result.children = [self, power_tensor]
        
        return result
    
    def relu(self):
        """ReLU activation function applied element-wise."""
        result_data = self._element_wise_op(self.data, None, lambda x, _: max(0, x))
        result = Tensor(result_data, self.shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                # Create a mask for the ReLU derivative
                mask_data = self._element_wise_op(self.data, None, lambda x, _: 1.0 if x > 0 else 0.0)
                mask = Tensor(mask_data, self.shape)
                
                self.backward(grad * mask)
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def sigmoid(self):
        """Sigmoid activation function applied element-wise."""
        result_data = self._element_wise_op(
            self.data, None, lambda x, _: 1.0 / (1.0 + math.exp(-x))
        )
        result = Tensor(result_data, self.shape, self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor([1.0])
                
                # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                sigmoid_derivative = result * (1 - result)
                self.backward(grad * sigmoid_derivative)
            
            result._grad_fn = _backward
            result.children = [self]
        
        return result
    
    def __repr__(self):
        """String representation of the tensor."""
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad})"