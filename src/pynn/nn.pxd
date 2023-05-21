from .core cimport GraphNode


cdef class Parameters:
    """网络参数。"""

    # 图节点
    cdef GraphNode _node


cdef class WeightParameters(Parameters): pass
cdef class BiasParameters(Parameters): pass


cdef class Layer:
    """网络层。"""

    cpdef tuple parameters(self) noexcept


cdef class LinearLayer(Layer):

    # 权重
    cdef WeightParameters _weight
    # 偏置
    cdef BiasParameters _bias


cdef class Optimizer:
    """优化器。"""

    cdef list _weights, _bias


cdef class SGD(Optimizer):
    """SGD 优化器。"""

    cdef float _alpha
