cimport numpy as cnp

from .core cimport GraphNode


cdef cnp.ndarray negfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray addfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray subfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray matmulfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray mulfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray divfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray powfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray Tfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray sumfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray sumfunc0(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray sumfunc1(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray expfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray logfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray relufunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept


cdef cnp.ndarray softmaxlossfunc(
    cnp.ndarray grad,
    GraphNode left,
    GraphNode right,
    bint flags,
) noexcept
