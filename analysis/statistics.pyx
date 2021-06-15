# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
""" Handles image masking operations """

# cython
cimport cython
from cython.operator cimport dereference as deref
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from analysis.statistics cimport StatsResults, compute_stats
from analysis.types cimport bool_t, uint8_t, uint32_t, int32_t, uint64_t, NPBOOL_t, NPUINT_t, NPINT32_t, NPUINT32_t, NPLONGLONG_t, NPFLOAT_t
import json
import math


cdef class Statistics(object):
    cdef StatsResults stats

    def __cinit__(self, uint64_t pointer):
        self.stats = deref(<StatsResults*> pointer)  # StatsResults() # construct new obj

    cpdef bool_t is_valid(self):
        return self.stats.is_valid()

    cpdef bool_t has_quartile(self):
        return self.stats.has_quartile()

    cpdef bool_t has_CV(self):
        return self.stats.mean > -0.001 and self.stats.mean < 0.001 

    cpdef bool_t is_nan(self, double val):
        return self.stats.is_nan(val)

    def get_sum(self):
        if self.stats.is_nan(self.stats.sum):
            return math.nan
        return float(self.stats.sum)

    def get_mean(self):
        if self.stats.is_nan(self.stats.mean):
            return math.nan
        return float(self.stats.mean)

    def get_variance(self):
        if self.stats.is_nan(self.stats.variance):
            return math.nan
        return float(self.stats.variance)

    def get_STD(self):
        if self.stats.is_nan(self.stats.STD):
            return math.nan
        return float(self.stats.STD)

    def get_CV(self):
        if self.stats.is_nan(self.stats.coeff_variation):
            return math.nan
        return float(self.stats.coeff_variation)

    def get_min(self):
        if self.stats.is_nan(self.stats.min):
            return math.nan
        return float(self.stats.min)

    def get_max(self):
        if self.stats.is_nan(self.stats.max):
            return math.nan
        return float(self.stats.max)

    def get_range(self):
        if self.stats.is_nan(self.stats.range):
            return math.nan
        return float(self.stats.range)

    def get_Q1(self):
        if self.stats.is_nan(self.stats.Q1):
            return math.nan
        return float(self.stats.Q1)

    def get_median(self):
        if self.stats.is_nan(self.stats.median):
            return math.nan
        return float(self.stats.median)

    def get_Q3(self):
        if self.stats.is_nan(self.stats.Q3):
            return math.nan
        return float(self.stats.Q3)

    def get_IQR(self):
        if self.stats.is_nan(self.stats.IQR):
            return math.nan
        return float(self.stats.IQR)

    def json(self):
        return {
            'sum': self.get_sum(),
            'mean': self.get_mean(),
            'variance': self.get_variance(),
            'STD': self.get_STD(),
            'CV': self.get_CV(),
            'min': self.get_min(),
            'max': self.get_max(),
            'range': self.get_range(),
            'Q1': self.get_Q1(),
            'median': self.get_median(),
            'Q3': self.get_Q3(),
            'IQR': self.get_IQR()
        }
    
    @staticmethod
    def get_header_data():
        return ['Sum', 'Mean', 'Variance', 'STD', 'CV', 'Min', 'Max', 'Range', 'Q1', 'Median', 'Q3', 'IQR']

    def get_row_data(self):
        return [self.get_sum(), self.get_mean(), self.get_variance(), self.get_STD(), self.get_CV(), self.get_min(), self.get_max(), self.get_range(), self.get_Q1(), self.get_median(), self.get_Q3(), self.get_IQR()]

    def get_str_row_data(self):
        return list(map(str, self.get_row_data()))

    def __str__(self):
        return json.dumps(
            self.json()
        )


cdef Statistics make_statistics(StatsResults *res):
    # copy the data over to the new python object
    stats = Statistics(<uint64_t> res)
    return stats


cpdef object compute_statistics(double[:] data):
    cdef vector[double] compute
    cdef StatsResults results
    cdef Statistics stats

    # add values from memoryview
    for d in data:
        compute.push_back(<double> d)
    
    # compute the results
    results = compute_stats(compute)
    stats = Statistics(<uint64_t> &results)

    return stats
