import gzip
import numpy as np

import pynn
from pynn.nn import LinearLayer, SGD
from pynn.loss import softmaxloss


def load_data() -> np.ndarray:
    """加载数据集。"""

    data = {
        'train_images': '../data/train-images-idx3-ubyte.gz',
        'train_lables': '../data/train-labels-idx1-ubyte.gz',
        'test_images': '../data/t10k-images-idx3-ubyte.gz',
        'test_labels': '../data/t10k-labels-idx1-ubyte.gz',
    }
    for k in data:
        path = data[k]
        with gzip.open(path, 'rb') as f:
            buf = f.read()
        if 'image' in k:
            data[k] = np.frombuffer(
                buf,
                np.uint8,
                offset=16
            ).reshape(-1, 28 * 28)
        else:
            data[k] = np.eye(10)[np.frombuffer(buf, np.uint8, offset=8)]
    return data


class NN:
    """网络。"""

    def __init__(self) -> None:
        self.layer1 = LinearLayer(28 * 28, 14 * 14)
        self.layer2 = LinearLayer(14 * 14, 7 * 7)
        self.layer3 = LinearLayer(49, 10)
        self.optimizer = SGD(*(
            self.layer1.parameters() +
            self.layer2.parameters() +
            self.layer3.parameters()
        ), alpha=0.00001)

    def __call__(self, X) -> pynn.GraphNode:
        X = self.layer1(X).relu()
        X = self.layer2(X).relu()
        X = self.layer3(X).relu()
        return X

    def step(self) -> None:
        self.optimizer.step()


data = load_data()
nn = NN()
for img, label in zip(data['train_images'], data['train_lables']):
    out = nn(img.reshape(-1, 1))
    loss = softmaxloss(out, label.reshape(-1, 1))
    loss.backward()
    print(out, label.reshape(-1, 1), sep='\n')
    print('loss:', loss)
    nn.step()
