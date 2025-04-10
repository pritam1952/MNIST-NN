# data.py
import struct
import os
import random
import urllib.request
import gzip
import shutil
import sys
from tensor import Tensor

def direct_download_mnist():
    """
    Download MNIST dataset files directly from a reliable source and extract them
    to the current directory.
    """
    os.makedirs('mnist_data', exist_ok=True)
    
    # Direct links to MNIST files (hosted on reliable servers)
    urls = [
        ('http://github.com/nlhkh/mnist-python/raw/master/data/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz'),
        ('http://github.com/nlhkh/mnist-python/raw/master/data/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
        ('http://github.com/nlhkh/mnist-python/raw/master/data/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz'),
        ('http://github.com/nlhkh/mnist-python/raw/master/data/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    ]
    
    # Download and extract each file
    for url, filename in urls:
        gz_path = os.path.join('mnist_data', filename)
        extract_path = os.path.join('mnist_data', filename[:-3])  # Remove .gz
        
        # Skip if already extracted
        if os.path.exists(extract_path):
            print(f"{extract_path} already exists, skipping download")
            continue
        
        # Download
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            # Try alternative URL if primary fails
            backup_url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{filename}"
            print(f"Trying backup URL: {backup_url}")
            try:
                urllib.request.urlretrieve(backup_url, gz_path)
            except Exception as e2:
                print(f"Error with backup URL: {e2}")
                print(f"Failed to download {filename}")
                continue
        
        # Extract
        print(f"Extracting {gz_path}...")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extract_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully extracted to {extract_path}")
        except Exception as e:
            print(f"Error extracting {gz_path}: {e}")
    
    # Check if all files were extracted successfully
    required_files = [
        'mnist_data/train-images-idx3-ubyte',
        'mnist_data/train-labels-idx1-ubyte',
        'mnist_data/t10k-images-idx3-ubyte',
        'mnist_data/t10k-labels-idx1-ubyte'
    ]
    
    all_files_exist = all(os.path.exists(f) for f in required_files)
    if all_files_exist:
        print("All MNIST files downloaded and extracted successfully!")
        return True
    else:
        missing = [f for f in required_files if not os.path.exists(f)]
        print(f"Missing files: {missing}")
        return False

def read_idx_file(file_path):
    """
    Read IDX file format (used by MNIST).
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic % 256
        
        # Read dimensions
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        
        # Read data
        size = 1
        for dim in shape:
            size *= dim
        
        data = []
        buffer = f.read(size)
        
        if ndim == 1:  # For labels
            return list(buffer)
        
        # For images
        offset = 0
        for i in range(shape[0]):
            image = []
            for j in range(shape[1]):
                row = []
                for k in range(shape[2]):
                    pixel = buffer[offset]
                    row.append(float(pixel) / 255.0)  # Normalize to [0, 1]
                    offset += 1
                image.append(row)
            data.append(image)
        
        return data

class MNISTDataset:
    def __init__(self, train=True):
        """
        Initialize the MNIST dataset.
        
        Parameters:
        train (bool): Whether to load training or test data
        """
        self.train = train
        
        # Ensure we have the dataset files
        if not os.path.exists('mnist_data'):
            print("MNIST dataset not found. Downloading...")
            if not direct_download_mnist():
                raise RuntimeError("Failed to download MNIST dataset")
        
        # File paths
        if train:
            images_file = 'mnist_data/train-images-idx3-ubyte'
            labels_file = 'mnist_data/train-labels-idx1-ubyte'
        else:
            images_file = 'mnist_data/t10k-images-idx3-ubyte'
            labels_file = 'mnist_data/t10k-labels-idx1-ubyte'
        
        # Load data
        try:
            print(f"Loading MNIST {'training' if train else 'test'} data...")
            self.images = read_idx_file(images_file)
            self.labels = read_idx_file(labels_file)
            print(f"Successfully loaded {len(self.images)} examples")
        except Exception as e:
            print(f"Error loading MNIST data: {e}")
            # If file exists but can't be read, it might be corrupted
            if os.path.exists(images_file) and os.path.exists(labels_file):
                print("Dataset files might be corrupted. Re-downloading...")
                # Delete existing files and re-download
                try:
                    for f in os.listdir('mnist_data'):
                        os.remove(os.path.join('mnist_data', f))
                except:
                    pass
                
                if not direct_download_mnist():
                    raise RuntimeError("Failed to re-download MNIST dataset")
                
                # Try loading again
                self.images = read_idx_file(images_file)
                self.labels = read_idx_file(labels_file)
            else:
                raise
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        image_tensor = Tensor(image, (28, 28))
        label_tensor = Tensor(label, ())
        
        return image_tensor, label_tensor

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            
            # Collect batch data
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                image, label = self.dataset[idx]
                # Flatten image for simplicity
                batch_images.append(image.flatten().data)
                batch_labels.append(label.data)
            
            # Create batch tensors
            images_tensor = Tensor(batch_images, (len(batch_indices), 28*28))
            labels_tensor = Tensor(batch_labels, (len(batch_indices),))
            
            yield images_tensor, labels_tensor
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Direct execution to download the dataset
if __name__ == "__main__":
    print("=== MNIST Dataset Downloader ===")
    try:
        # Force re-download if requested
        if len(sys.argv) > 1 and sys.argv[1] == '--force':
            print("Forcing re-download of MNIST dataset...")
            if os.path.exists('mnist_data'):
                for f in os.listdir('mnist_data'):
                    try:
                        os.remove(os.path.join('mnist_data', f))
                    except:
                        pass
        
        # Download the dataset
        direct_download_mnist()
        
        # Test loading
        print("\nTesting dataset loading...")
        train_dataset = MNISTDataset(train=True)
        test_dataset = MNISTDataset(train=False)
        
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of test examples: {len(test_dataset)}")
        
        print("\nLoading successful! You can now use the MNIST dataset in your code.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you're still having issues, please try running with the --force flag:")
        print("python data.py --force")