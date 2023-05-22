import numpy as np

import pynn
from pynn.nn import LinearLayer, SGD
from pynn.loss import mseloss


class NN:
    """网络。"""

    def __init__(self) -> None:
        self.layer1 = LinearLayer(2, 2)
        self.layer2 = LinearLayer(2, 1)
        self.optimizer = SGD(*(
            self.layer1.parameters() +
            self.layer2.parameters()
        ), alpha=0.001)

    def __call__(self, X) -> pynn.GraphNode:
        X = self.layer1(X)
        X = self.layer2(X)
        return X

    def step(self) -> None:
        self.optimizer.step()


data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
])
labels = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
])
nn = NN()
for _ in range(3000):
    for d, label in zip(data, labels):
        out = nn(d.reshape(-1, 1))
        loss = mseloss(out, label)
        loss.backward()
        nn.step()
        print(loss)
print(nn(data[0].reshape(-1, 1)))
print(nn(data[1].reshape(-1, 1)))
print(nn(data[2].reshape(-1, 1)))
print(nn(data[3].reshape(-1, 1)))
