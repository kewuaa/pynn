# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: language_level=3
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from numpy.math cimport logf
from cpython.mem cimport PyMem_Malloc
cimport numpy as cnp

from .core cimport GraphNode


cdef cnp.ndarray negfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """负号求导函数。"""

    return -grad


cdef cnp.ndarray addfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """相加求导函数。"""

    return grad


cdef cnp.ndarray subfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """相减求导函数。"""

    if flags:
        return -grad
    else:
        return grad


cdef cnp.ndarray matmulfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """矩阵乘法求导函数。"""

    if flags:
        return left._tensor.T @ grad
    else:
        return grad @ right._tensor.T


cdef cnp.ndarray mulfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """相乘求导函数。"""

    if flags:
        return grad * left._tensor
    else:
        return grad * right._tensor


cdef cnp.ndarray divfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """相除求导函数。"""

    if flags:
        return -left._tensor / (right._tensor ** 2) * grad
    else:
        return grad / right._tensor


cdef cnp.ndarray powfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """幂函数求导。"""

    cdef float power = right._value
    return grad * power * (left._tensor ** (power - 1))


cdef cnp.ndarray Tfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """转置求导函数。"""

    return grad.T


cdef cnp.ndarray sumfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """求和求导函数。"""

    cdef float* arr = <float*>PyMem_Malloc(sizeof(float) * left._size)
    cdef float[::1] view = <float[:left._size]>arr
    view[:] = 1.
    cdef cnp.ndarray _grad = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        left._ndim,
        left._shape,
        cnp.NPY_FLOAT32,
        <void*>arr
    )
    cnp.PyArray_ENABLEFLAGS(_grad, cnp.NPY_ARRAY_OWNDATA)
    return _grad * grad


cdef cnp.ndarray sumfunc0(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """0轴求和求导函数。"""

    cdef float* arr = <float*>PyMem_Malloc(sizeof(float) * left._shape[0])
    cdef float[::1] view = <float[:left._shape[0]]>arr
    cdef Py_ssize_t shape[2]
    shape[0] = left._shape[0]
    shape[1] = 1
    view[:] = 1.
    cdef cnp.ndarray _grad = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, shape, cnp.NPY_FLOAT32, <void*>arr
    )
    cnp.PyArray_ENABLEFLAGS(_grad, cnp.NPY_ARRAY_OWNDATA)
    return _grad @ grad


cdef cnp.ndarray sumfunc1(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """1轴求和求导函数。"""

    cdef float* arr = <float*>PyMem_Malloc(sizeof(float) * left._shape[1])
    cdef float[::1] view = <float[:left._shape[1]]>arr
    cdef Py_ssize_t shape[2]
    view[:] = 1.
    shape[0] = 1
    shape[1] = left._shape[1]
    cdef cnp.ndarray _grad = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, shape, cnp.NPY_FLOAT32, <void*>arr
    )
    cnp.PyArray_ENABLEFLAGS(_grad, cnp.NPY_ARRAY_OWNDATA)
    return grad @ _grad


cdef cnp.ndarray expfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """指数函数求导函数。"""

    grad = grad * left._tensor
    if right._value == 0.:
        return grad
    else:
        return grad * logf(right._value)


cdef cnp.ndarray logfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """log 求导函数。"""

    grad = grad / left._tensor
    if right._value == 0.:
        return grad
    else:
        return grad / logf(right._value)


cdef cnp.ndarray relufunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """relu 函数求导函数。"""

    cdef cnp.ndarray _grad = cnp.PyArray_ZEROS(
        left._ndim,
        left._shape,
        cnp.NPY_INT8,
        0
    )
    _grad[left._tensor > 0] = 1
    return grad * _grad


cdef cnp.ndarray mselossfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """mse loss 求导函数。"""

    return grad * right._tensor


cdef cnp.ndarray softmaxlossfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept:
    """softmax loss 求导函数。"""

    return grad * right._tensor
