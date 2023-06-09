# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: language_level=3
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cpython.mem cimport Py_buffer, PyMem_Malloc, PyMem_Free
from numpy.math cimport logf
cimport numpy as cnp
import numpy as np

from .gradfunc cimport (
    negfunc,
    addfunc,
    subfunc,
    matmulfunc,
    mulfunc,
    divfunc,
    powfunc,
    expfunc,
    Tfunc,
    sumfunc,
    sumfunc0,
    sumfunc1,
    relufunc,
)
cnp.import_array()
cdef cnp.ndarray _initial_grad = _initialize_grad()


cdef inline GraphNode _to_graph_node(tensor) noexcept:
    return tensor if isinstance(tensor, GraphNode) else GraphNode(tensor)


cdef cnp.ndarray _initialize_grad() noexcept:
    cdef float* arr = <float*>PyMem_Malloc(sizeof(float))
    cdef Py_ssize_t[2] shape = [1, 1]
    arr[0] = 1.
    cdef cnp.ndarray grad = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, shape, cnp.NPY_FLOAT32, <void*>arr
    )
    cnp.PyArray_ENABLEFLAGS(grad, cnp.NPY_ARRAY_OWNDATA)
    return grad


cdef class GraphNode:
    """反向图节点。"""

    def __init__(
        self,
        tensor,
        *,
        bint requires_grad=0
    ) -> None:
        """初始化。

        :param tensor: 输入的张量
        :param requires_grad: 是否需要计算梯度
        """

        cdef cnp.ndarray _tensor
        if isinstance(tensor, cnp.ndarray):
            _tensor = tensor
        elif isinstance(tensor, GraphNode):
            _tensor = tensor._tensor
        else:
            _tensor = np.asarray(tensor)
        self._tensor = _tensor
        self._grad = <cnp.ndarray>cnp.PyArray_ZEROS(
            cnp.PyArray_NDIM(_tensor),
            cnp.PyArray_DIMS(_tensor),
            cnp.NPY_FLOAT64, 0
        )
        self._save_grad = 0
        self._gradfunc = NULL
        self._as_unique = 0
        self._is_leaf = 1
        self._parent = None
        self._subnode_l = self._subnode_r = None
        self._opera_type = NULL
        if cnp.PyArray_NDIM(_tensor) > 2 and requires_grad:
            raise TypeError(f'{self._ndim} dim array could require grad')
        self.requires_grad = requires_grad

    cdef int _reset_grad(self) noexcept:
        self._grad[...] = 0.
        return 0

    cdef cnp.ndarray _get_grad(self) noexcept:
        """获取梯度值。"""

        cdef cnp.ndarray grad = cnp.PyArray_Copy(self._grad)
        if not self._save_grad:
            self._reset_grad()
        return grad

    @staticmethod
    cdef GraphNode _simple_value_node(float value) noexcept:
        """生成幂指数节点。"""

        cdef GraphNode node = GraphNode.__new__(GraphNode)
        node._value = value
        return node

    def __str__(self) -> str:
        return str(np.asarray(self._tensor))

    def __repr__(self) -> str:
        return '\n'.join((
            "----------------------------------------",
            repr(np.asarray(self._tensor)),
            f'grad: {self._grad}',
            f'operation: {self.opera_type}',
            f'requires grad: {self.requires_grad}',
            f'is leaf: {self._is_leaf}',
            "----------------------------------------",
        ))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """实现缓冲区协议。

        :param buffer: 缓冲区对象
        """

        cdef cnp.ndarray tensor = self._tensor
        cdef cnp.dtype dtype = <cnp.dtype>cnp.PyArray_DTYPE(tensor)
        buffer.obj = self
        buffer.buf = cnp.PyArray_DATA(tensor)
        buffer.format = dtype.char
        buffer.itemsize = dtype.itemsize
        buffer.len = cnp.PyArray_SIZE(tensor)
        buffer.ndim = cnp.PyArray_NDIM(tensor)
        buffer.shape = cnp.PyArray_DIMS(tensor)
        buffer.strides = cnp.PyArray_STRIDES(tensor)
        buffer.readonly = 0
        buffer.suboffsets = NULL

    def __getattr__(self, name: str):
        return self._tensor.__getattribute__(name)

    def __getitem__(self, index: int):
        return self._tensor[index]

    def __pos__(self) -> GraphNode:
        """正号。"""

        return self

    def __neg__(self) -> GraphNode:
        """负号。"""

        cdef GraphNode new_node = GraphNode(
            -self._tensor,
            requires_grad=self.requires_grad
        )
        new_node._gradfunc = negfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._opera_type = 'negative'
        self._parent = new_node
        return new_node

    def __add__(self, other) -> GraphNode:
        """加法。"""

        if isinstance(other, (int, float, long)):
            self._tensor = self._tensor + other
            return self
        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            self._tensor + _other._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = addfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = _other
        new_node._opera_type = 'add'
        self._parent = _other._parent = new_node
        return new_node

    def __radd__(self, other) -> GraphNode:
        """右加。"""

        return self.__add__(other)

    def __sub__(self, other) -> GraphNode:
        """减法。"""

        if isinstance(other, (int, float, long)):
            self._tensor = self._tensor - other
            return self
        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            self._tensor - _other._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = subfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = _other
        new_node._opera_type = 'subtract'
        self._parent = _other._parent = new_node
        return new_node

    def __rsub__(self, other) -> GraphNode:
        """右减。"""

        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            _other._tensor - self._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = subfunc
        new_node._is_leaf = 0
        new_node._subnode_l = _other
        new_node._subnode_r = self
        new_node._opera_type = 'reverse subtract'
        self._parent = _other._parent = new_node
        return new_node

    cpdef GraphNode matmul(self, other, bint reverse=0) noexcept:
        """矩阵乘法。"""

        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        if reverse:
            new_node = GraphNode(
                _other._tensor @ self._tensor,
                requires_grad=self.requires_grad or _other.requires_grad
            )
            new_node._subnode_l = _other
            new_node._subnode_r = self
        else:
            new_node = GraphNode(
                self._tensor @ _other._tensor,
                requires_grad=self.requires_grad or _other.requires_grad
            )
            new_node._subnode_l = self
            new_node._subnode_r = _other
        new_node._gradfunc = matmulfunc
        new_node._is_leaf = 0
        new_node._opera_type = 'matrix multiply'
        self._parent = _other._parent = new_node
        return new_node

    def __mul__(self, other) -> GraphNode:
        """乘法。"""

        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            self._tensor * _other._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = mulfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = _other
        new_node._opera_type = 'multiply'
        self._parent = _other._parent = new_node
        return new_node

    def __rmul__(self, other) -> GraphNode:
        """右乘。"""

        return self.__mul__(other)

    def __truediv__(self, other) -> GraphNode:
        """除法。"""

        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            self._tensor / _other._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = divfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = _other
        new_node._opera_type = 'division'
        self._parent = _other._parent = new_node
        return new_node

    def __rtruediv__(self, other) -> GraphNode:
        """右除。"""

        cdef GraphNode _other, new_node
        _other = _to_graph_node(other)
        new_node = GraphNode(
            _other._tensor / self._tensor,
            requires_grad=self.requires_grad or _other.requires_grad
        )
        new_node._gradfunc = divfunc
        new_node._is_leaf = 0
        new_node._subnode_l = _other
        new_node._subnode_r = self
        new_node._opera_type = 'reverse division'
        self._parent = _other._parent = new_node
        return new_node

    def __pow__(self, float power) -> GraphNode:
        """幂函数。"""

        cdef GraphNode new_node
        new_node = GraphNode(
            self._tensor ** power,
            requires_grad=self.requires_grad
        )
        new_node._gradfunc = powfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = GraphNode._simple_value_node(power)
        new_node._opera_type = 'pow'
        self._parent = new_node
        return new_node

    def __rpow__(self, float base) -> GraphNode:
        """指数函数。"""

        cdef GraphNode new_node = GraphNode(
            np.exp(self._tensor * logf(base)),
            requires_grad=self.requires_grad
        )
        new_node._gradfunc = expfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._subnode_r = GraphNode._simple_value_node(base)
        new_node._opera_type = 'exp'
        self._parent = new_node
        return new_node

    cpdef GraphNode transpose(self):
        """转置。"""

        cdef GraphNode new_node = GraphNode(
            self._tensor.T,
            requires_grad=self.requires_grad
        )
        new_node._gradfunc = Tfunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._opera_type = 'transpose'
        self._parent = new_node
        return new_node

    @property
    def T(self) -> GraphNode:
        return self.transpose()

    cpdef GraphNode sum(self, int axis=-1) noexcept:
        """求和。"""

        cdef GraphNode new_node = GraphNode(
            <double>self._tensor.sum(keepdims=1) if axis < 0
            else self._tensor.sum(axis=axis, keepdims=1),
            requires_grad=self.requires_grad
        )
        if axis < 0:
            new_node._gradfunc = sumfunc
            new_node._opera_type = 'sum'
        elif axis == 0:
            new_node._gradfunc = sumfunc0
            new_node._opera_type = 'sum_axis0'
        else:
            new_node._gradfunc = sumfunc1
            new_node._opera_type = 'sum_axis1'
        new_node._is_leaf = 0
        new_node._subnode_l = self
        self._parent = new_node
        return new_node

    cpdef GraphNode relu(self) noexcept:
        """relu 函数。"""

        cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_Copy(self._tensor)
        tensor[tensor < 0] = 0
        cdef GraphNode new_node = GraphNode(
            tensor,
            requires_grad=self.requires_grad
        )
        new_node._gradfunc = relufunc
        new_node._is_leaf = 0
        new_node._subnode_l = self
        new_node._opera_type = 'relu'
        self._parent = new_node
        return new_node

    cdef bint _same_shape(self, cnp.ndarray other) noexcept:
        """比较形状是否相同。

        :param other: 需要进行比较的对象
        """

        cdef unsigned int i
        cdef cnp.ndarray tensor = self._tensor
        cdef Py_ssize_t* self_shape = cnp.PyArray_DIMS(tensor)
        cdef Py_ssize_t* other_shape = cnp.PyArray_DIMS(other)
        for i in range(cnp.PyArray_NDIM(tensor)):
            if self_shape[i] != other_shape[i]:
                return 0
        return 1

    cdef int _backward(self, cnp.ndarray grad=_initial_grad) except 1:
        """反向传播。

        :param grad: 后一级传回的梯度
        """

        cdef cnp.ndarray _grad
        # print(self.opera_type)
        if self._as_unique or self._is_leaf:
            if not self._same_shape(grad):
                raise RuntimeError(
                    'not same shape: '
                    f'({self.shape[0]}, {self.shape[1]}) '
                    f'and ({grad.shape[0]}, {grad.shape[1]})'
                )
            self._grad += grad
        else:
            if self._subnode_l.requires_grad:
                _grad = self._gradfunc(
                    grad, self._subnode_l,
                    self._subnode_r, 0
                )
                self._subnode_l._backward(_grad)
            if self._subnode_r is not None and self._subnode_r.requires_grad:
                _grad = self._gradfunc(
                    grad, self._subnode_l,
                    self._subnode_r, 1
                )
                self._subnode_r._backward(_grad)
        return 0

    def backward(self) -> None:
        """反向传播的外部接口。"""

        if self._gradfunc is NULL:
            raise RuntimeError('could not backward for gradfunc is NULL')
        if cnp.PyArray_SIZE(self._tensor) != 1:
            raise RuntimeError('only scalar could backward')
        self._backward()

    @property
    def opera_type(self) -> str:
        if self._opera_type is NULL:
            return 'NULL'
        cdef str opera_type = (<bytes>self._opera_type).decode()
        return opera_type

    @property
    def grad(self) -> np.ndarray:
        """获取梯度值。"""

        return self._get_grad()

    def enable_grad(self) -> None:
        """允许计算梯度。"""

        self.requires_grad = 1

    def detach(self) -> None:
        """将该节点作为整体，其子节点全部脱离节点图。"""

        self._as_unique = 1

    def reshape(self, *shape):
        cdef cnp.ndarray tensor = self._tensor.reshape(*shape)
        return GraphNode(tensor)


cpdef GraphNode zeros(shape):
    cdef Py_ssize_t ndim = len(shape)
    cdef Py_ssize_t* _shape = <Py_ssize_t*>PyMem_Malloc(sizeof(Py_ssize_t) * ndim)
    cdef unsigned int i
    for i in range(ndim):
        _shape[i] = shape[i]
    cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_ZEROS(
        ndim, _shape,
        cnp.NPY_INT8, 0
    )
    PyMem_Free(_shape)
    return GraphNode(tensor)


cpdef GraphNode ones(shape):
    cdef Py_ssize_t ndim = len(shape)
    cdef Py_ssize_t* _shape = <Py_ssize_t*>PyMem_Malloc(sizeof(Py_ssize_t) * ndim)
    cdef Py_ssize_t size = 1
    cdef unsigned int i
    for i in range(ndim):
        _shape[i] = shape[i]
        size *= _shape[i]
    cdef int* data = <int*>PyMem_Malloc(sizeof(int) * size)
    cdef int[::1] view = <int[:size]>data
    view[:] = 1
    cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        ndim, _shape,
        cnp.NPY_INT32, <void*>data
    )
    cnp.PyArray_ENABLEFLAGS(tensor, cnp.NPY_ARRAY_OWNDATA)
    PyMem_Free(_shape)
    return GraphNode(tensor)


cpdef GraphNode empty(shape):
    cdef Py_ssize_t ndim = len(shape)
    cdef Py_ssize_t* _shape = <Py_ssize_t*>PyMem_Malloc(sizeof(Py_ssize_t) * ndim)
    cdef unsigned int i
    for i in range(ndim):
        _shape[i] = shape[i]
    cdef cnp.ndarray tensor = <cnp.ndarray>cnp.PyArray_EMPTY(
        ndim, _shape,
        cnp.NPY_FLOAT32, 0
    )
    PyMem_Free(_shape)
    return GraphNode(tensor)
