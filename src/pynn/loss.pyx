# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from numpy.math cimport logf
cimport numpy as cnp
import numpy as np

from .core cimport GraphNode
from .gradfunc cimport mselossfunc, softmaxlossfunc


cdef cnp.ndarray _to_tensor(array):
    if isinstance(array, cnp.ndarray):
        return array
    elif isinstance(array, GraphNode):
        return array._tensor
    else:
        return np.asarray(array)


cpdef GraphNode mseloss(GraphNode X, Y):
    """mse loss 函数。

    :param X: 预测值
    :param Y: 真实值
    """

    cdef cnp.ndarray _Y = _to_tensor(Y)
    cdef GraphNode new_node = GraphNode(
        ((X._tensor - _Y) ** 2).sum(),
        requires_grad=X.requires_grad
    )
    new_node._gradfunc = mselossfunc
    new_node._is_leaf = 0
    new_node._subnode_l = X
    new_node._subnode_r = GraphNode.__new__(GraphNode)
    new_node._subnode_r._tensor = 2 * (X._tensor - _Y)
    new_node._opera_type = 'mse loss'
    X._parent = new_node
    return new_node


cpdef GraphNode softmaxloss(GraphNode X, Y):
    """softmax loss 函数。

    :param X: 预测值，为列向量
    :param Y: onehot 标签， 为列向量
    """

    cdef cnp.ndarray _Y = _to_tensor(Y)
    cdef cnp.ndarray _X = X._tensor - X._tensor.max()
    cdef cnp.ndarray expx = np.exp(_X)
    cdef cnp.ndarray softmaxx = expx / expx.sum()
    cdef GraphNode new_node = GraphNode(
        _Y.T @ (logf(expx.sum()) - _X),
        requires_grad=X.requires_grad
    )
    new_node._gradfunc = softmaxlossfunc
    new_node._is_leaf = 0
    new_node._subnode_l = X
    new_node._subnode_r = GraphNode.__new__(GraphNode)
    new_node._subnode_r._tensor = softmaxx - _Y
    new_node._opera_type = 'softmax loss'
    X._parent = new_node
    return new_node
