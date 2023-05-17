from numpy.math cimport logf
cimport numpy as cnp
import numpy as np

from .core cimport GraphNode, Array
from .gradfunc cimport expfunc, logfunc, softmaxlossfunc


cpdef GraphNode transpose(GraphNode node):
    """转置。

    :param node: 输入图节点
    """

    return node.transpose()


cpdef GraphNode exp(GraphNode node):
    """自然底数指数函数。

    :param node: 输入图节点
    """

    cdef GraphNode new_node = GraphNode(
        np.exp(node._tensor),
        requires_grad=node.requires_grad
    )
    new_node._gradfunc = expfunc
    new_node._is_leaf = 0
    new_node._subnode_l = node
    new_node._subnode_r = GraphNode._simple_value_node(0.)
    new_node._opera_type = 'nature base exp'
    node._parent = new_node
    return new_node


cpdef GraphNode log(GraphNode node, float base=0.):
    """log 函数

    :param node: 输入图节点
    :param base: 底数
    """

    if base < 0:
        raise ValueError('base must be positive')
    cdef cnp.ndarray tensor
    if base == 0.:
        tensor = np.log(node._tensor)
    elif base == 2.:
        tensor = np.log2(node._tensor)
    elif base == 10.:
        tensor = np.log10(node._tensor)
    else:
        tensor = np.log(node._tensor) / logf(base),
    cdef GraphNode new_node = GraphNode(
        tensor,
        requires_grad=node.requires_grad,
    )
    new_node._gradfunc = logfunc
    new_node._is_leaf = 0
    new_node._subnode_l = node
    new_node._subnode_r = GraphNode._simple_value_node(base)
    new_node._opera_type = 'log'
    node._parent = new_node
    return new_node


cpdef GraphNode relu(GraphNode node):
    """relu 函数。"""

    return node.relu()


cpdef GraphNode softmaxloss(GraphNode X, Y):
    """softmax loss 函数。

    :param X: 预测值，为列向量
    :param Y: onehot 标签， 为列向量
    """

    cdef cnp.ndarray _Y
    if isinstance(Y, cnp.ndarray):
        _Y = Y
    elif isinstance(Y, GraphNode):
        _Y = Y._tensor
    else:
        _Y = np.asarray(Y)
    cdef cnp.ndarray expx = np.exp(X._tensor)
    cdef cnp.ndarray softmaxx = expx / expx.sum()
    cdef GraphNode new_node = GraphNode(
        _Y.T @ (logf(expx.sum()) - X._tensor),
        requires_grad=X.requires_grad
    )
    new_node._gradfunc = softmaxlossfunc
    new_node._is_leaf = 0
    new_node._subnode_l = X
    new_node._subnode_r = GraphNode.__new__(GraphNode)
    new_node._subnode_r._tensor = softmaxx - _Y
    new_node._opera_type = 'softmaxloss'
    X._parent = new_node
    return new_node
