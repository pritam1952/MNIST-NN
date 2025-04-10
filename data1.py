# data.py
import struct
import os
import random
from tensor import Tensor

def read_idx_file(file_path):
    """Read IDX file format (used by MNIST)."""
    with open(file_path, 'rb') as f:
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
    def __init__(self, root='./data', train=True, use_random=True, size=60000):
        self.root = root
        self.train = train
        self.use_random = use_random
        self.size = size  # Number of samples for random data

        if self.use_random:
            # Generate random data
            self.images = [
                [[random.random() for _ in range(28)] for _ in range(28)]
                for _ in range(self.size)
            ]
            self.labels = [random.randint(0, 9) for _ in range(self.size)]
        else:
            # File paths
            if train:
                images_file = os.path.join(root, 'train-images-idx3-ubyte')
                labels_file = os.path.join(root, 'train-labels-idx1-ubyte')
            else:
                images_file = os.path.join(root, 't10k-images-idx3-ubyte')
                labels_file = os.path.join(root, 't10k-labels-idx1-ubyte')

            # Load data from files
            self.images = read_idx_file(images_file)
            self.labels = read_idx_file(labels_file)

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
            batch_indices = self.indices[i:i + self.batch_size]

            # Collect batch data
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                image, label = self.dataset[idx]
                # Flatten image for simplicity
                batch_images.append(image.flatten().data)
                batch_labels.append(label.data)

            # Create batch tensors
            images_tensor = Tensor(batch_images, (len(batch_indices), 28 * 28))
            labels_tensor = Tensor(batch_labels, (len(batch_indices),))

            yield images_tensor, labels_tensor

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
