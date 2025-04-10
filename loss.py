# loss.py
from tensor import Tensor
import math

class Loss:
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

class MSELoss(Loss):
    def forward(self, predictions, targets):
        """Mean Squared Error Loss."""
        assert predictions.shape == targets.shape, "Predictions and targets shapes must match"
        diff = predictions - targets
        squared = diff * diff
        return squared.mean()

class CrossEntropyLoss(Loss):
    def forward(self, logits, targets):
        """Cross Entropy Loss for classification tasks."""
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Apply softmax to the logits
        max_vals = []
        for i in range(batch_size):
            max_vals.append(max(logits.data[i]))
        
        # Compute exp(logits - max) for numerical stability
        exp_logits = []
        for i in range(batch_size):
            row = []
            for j in range(num_classes):
                row.append(math.exp(logits.data[i][j] - max_vals[i]))
            exp_logits.append(row)
        
        # Compute softmax values
        softmax_probs = []
        for i in range(batch_size):
            row_sum = sum(exp_logits[i])
            row = [exp_val / row_sum for exp_val in exp_logits[i]]
            softmax_probs.append(row)
        
        # Calculate loss
        total_loss = 0.0
        for i in range(batch_size):
            target_idx = int(targets.data[i])
            assert 0 <= target_idx < num_classes, f"Target index {target_idx} out of range"
            prob = softmax_probs[i][target_idx]
            # Add a small epsilon to avoid log(0)
            epsilon = 1e-15
            total_loss -= math.log(max(prob, epsilon))
        
        loss = total_loss / batch_size
        
        # Create a tensor with the loss and set up backward pass
        loss_tensor = Tensor(loss, (), logits.requires_grad)
        
        if logits.requires_grad:
            def _backward(grad=None):
                if grad is None:
                    grad = Tensor(1.0)
                
                # Gradient of cross entropy with respect to logits
                grad_data = []
                for i in range(batch_size):
                    row_grad = []
                    target_idx = int(targets.data[i])
                    for j in range(num_classes):
                        # -1/batch_size for target class, softmax_prob/batch_size for others
                        if j == target_idx:
                            row_grad.append((-1.0 + softmax_probs[i][j]) / batch_size)
                        else:
                            row_grad.append(softmax_probs[i][j] / batch_size)
                    grad_data.append(row_grad)
                
                logits.backward(Tensor(grad_data, logits.shape) * grad)
            
            loss_tensor._grad_fn = _backward
            loss_tensor.children = [logits]
        
        return loss_tensor