# main.py
from tensor import Tensor
from nn import Module
from model import MNISTNet
from loss import CrossEntropyLoss
from optim import SGD
from data import MNISTDataset, DataLoader
import random
import math

def train(model, train_loader, criterion, optimizer, epochs=5):
    """Train the model."""
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            preds = []
            for i in range(outputs.shape[0]):
                preds.append(outputs.data[i].index(max(outputs.data[i])))
            
            # Count correct predictions
            for i in range(len(preds)):
                if int(preds[i]) == int(targets.data[i]):
                    correct += 1
            total += len(targets.data)
            
            # Print batch results
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.data:.4f} | Acc: {correct/total:.4f}")
            
            total_loss += loss.data
        
        # Print epoch results
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch: {epoch+1}/{epochs} complete | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    
    print("Training complete!")

def evaluate(model, test_loader):
    """Evaluate the model."""
    print("Evaluating model...")
    model.eval()
    
    correct = 0
    total = 0
    
    for data, targets in test_loader:
        # Forward pass
        outputs = model(data)
        
        # Get predictions
        preds = []
        for i in range(outputs.shape[0]):
            preds.append(outputs.data[i].index(max(outputs.data[i])))
        
        # Count correct predictions
        for i in range(len(preds)):
            if int(preds[i]) == int(targets.data[i]):
                correct += 1
        total += len(targets.data)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    epochs = 5
    
    # Load data
    print("Loading datasets...")
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    print("Initializing model...")
    model = MNISTNet()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, epochs)
    
    # Evaluate the model
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()