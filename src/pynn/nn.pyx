# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cimport numpy as cnp


cdef class Parameters: pass


cdef class WeightParameters(Parameters):
    """权重。"""

    def __init__(self, unsigned int row, unsigned int col) -> None:
        pass


cdef class BiasParameters(Parameters):
    """偏置。"""

    def __init__(self, unsigned int )
