# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cpython.mem cimport PyMem_Malloc
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
cimport numpy as cnp

from .core cimport GraphNode


cdef class WeightParameters(Parameters):
    """权重。"""

    def __init__(self, unsigned int row, unsigned int col) -> None:
        cdef unsigned int size = row * col
        cdef float* data = <float*>PyMem_Malloc(sizeof(float) * size)
        cdef float[::1] view = <float[:size]>data
        srand(<unsigned int>time(NULL))
        cdef unsigned int i
        for i in range(size):
            view[i] = rand() / (RAND_MAX + 1.)
        cdef Py_ssize_t[2] shape = [row, col]
        cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
            2, shape, cnp.NPY_FLOAT32, <void*>data
        )
        cnp.PyArray_ENABLEFLAGS(tensor, cnp.NPY_ARRAY_OWNDATA)
        self._node = GraphNode(tensor, requires_grad=1)


cdef class BiasParameters(Parameters):
    """偏置。"""

    def __init__(self, unsigned int row) -> None:
        cdef Py_ssize_t[2] shape = [row, 1]
        cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_ZEROS(
            2, shape, cnp.NPY_FLOAT32, 0
        )
        self._node = GraphNode(tensor, requires_grad=1)


cdef class Layer:
    """网络层。"""

    cpdef tuple parameters(self) noexcept:
        """返回层中的参数。"""

        pass


cdef class LinearLayer(Layer):
    """线性层。"""

    def __init__(
        self,
        unsigned int input_num,
        unsigned int output_num
    ) -> None:
        """初始化。

        :param input_num: 输入特征数
        :param output_num: 输出特征数
        """

        self._weight = WeightParameters(output_num, input_num)
        self._bias = BiasParameters(output_num)

    def __call__(self, X):
        """前向计算。"""

        return self._weight._node.matmul(X) + self._bias._node

    cpdef tuple parameters(self) noexcept:
        return self._weight, self._bias


cdef class SGD(Optimizer):
    """SGD 优化器。"""

    def __init__(self, *parameters, double alpha=0.001):
        """初始化。

        :param parameters: 所有参数
        """

        cdef unsigned int num = len(parameters)
        cdef unsigned int i
        self._weights = []
        self._bias = []
        for i in range(num):
            if isinstance(parameters[i], WeightParameters):
                self._weights.append(parameters[i])
            else:
                self._bias.append(parameters[i])
        self._alpha = alpha

    def step(self) -> None:
        """更新参数。"""

        cdef unsigned int i, num
        cdef Parameters param
        cdef cnp.ndarray grad
        num = len(self._weights)
        for i in range(num):
            param = self._weights[i]
            grad = param._node._get_grad()
            param._node._tensor -= self._alpha * grad
        num = len(self._bias)
        for i in range(num):
            param = self._bias[i]
            grad = param._node._get_grad()
            param._node._tensor -= self._alpha * grad
