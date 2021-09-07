import os
import numpy as np
from torch.utils.data import Dataset
import gzip


class Mnist(Dataset):
    def __init__(self, root, train=True, transform=None):

        # The file name prefix is obtained according to whether it is a training set
        self.file_pre = 'train' if train == True else 't10k'
        self.transform = transform

        # Generate the image and label file path of the corresponding dataset
        self.label_path = os.path.join(root,
                                       '%s-labels-idx1-ubyte.gz' % self.file_pre)
        self.image_path = os.path.join(root,
                                       '%s-images-idx3-ubyte.gz' % self.file_pre)

        # Read file data and return pictures and labels
        self.images, self.labels = self.__read_data__(
            self.image_path,
            self.label_path)

    def __read_data__(self, image_path, label_path):
        # Data set read
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8,
                                   offset=8)
        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8,
                                   offset=16).reshape(len(labels), 28, 28)
        return images, labels

    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])

        # If you need to convert to tensor, use tansform
        if self.transform is not None:
            image = self.transform(np.array(image))  # Need to use here np.array(image)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # Generate instance
    train_set = Mnist(
        root=r'C:\Users\young\PycharmProjects\pythonProject\LeNet-5\MNIST_data',
        train=False,
    )

    # Take a set of data and display it
    (data, label) = train_set[0]
    import matplotlib.pyplot as plt
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.title('label is :{}'.format(label))
    plt.show()
