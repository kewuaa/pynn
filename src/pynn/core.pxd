cimport numpy as cnp


ctypedef char* Operation


cdef class GraphNode:
    """反向图节点。"""

    cdef:
        # 张量数据
        cnp.ndarray _tensor
        # 维度
        Py_ssize_t _ndim
        # 形状
        Py_ssize_t* _shape
        # 类型
        cnp.dtype _dtype
        # 大小
        Py_ssize_t _size
        # 梯度值
        cnp.ndarray _grad
        # 是否保存梯度值
        bint _save_grad
        # 反向求导函数
        cnp.ndarray (*_gradfunc)(cnp.ndarray, GraphNode, GraphNode, bint) noexcept
        # 运算类型
        Operation _opera_type
        # 是否作为一个整体，不继续反向传播
        bint _as_unique
        # 是否为叶子节点
        bint _is_leaf
        # 父节点
        GraphNode _parent
        # 左右节点
        GraphNode _subnode_l, _subnode_r
        # 兼容幂指数，底数的值
        float _value

    cdef readonly:
        # 是否需要计算梯度
        bint requires_grad

    @staticmethod
    cdef GraphNode _simple_value_node(float value) noexcept
    cdef int _update_grad(self) noexcept
    cpdef GraphNode matmul(self, other) noexcept
    cpdef GraphNode transpose(self) noexcept
    cpdef GraphNode sum(self, int axis=*) noexcept
    cpdef GraphNode relu(self) noexcept
    cdef bint _same_shape(self, cnp.ndarray other) noexcept
    cdef int _backward(self, cnp.ndarray grad=*) except 1


cpdef GraphNode zeros(shape)
cpdef GraphNode ones(shape)
cpdef GraphNode empty(shape)
