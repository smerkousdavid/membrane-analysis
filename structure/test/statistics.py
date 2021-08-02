import unittest
from analysis.statistics import compute_statistics
import numpy as np
import math

NAN = math.nan
MAX_DIFF = 0.001  # max difference in numbers

def compare_statistics(data, res):
    # compare each key
    for key in res.keys():
        if key not in data:
            raise ValueError('key of ' + key + ' is not in returned data')

        if math.isnan(res[key]) and not math.isnan(data[key]):
            raise ValueError('key of ' + key + ' in dataset is not NaN when it was supposed to be EXPECTED: ' + str(res) + ' and RESULTS: ' + str(data))
        elif not math.isnan(res[key]) and math.isnan(data[key]):
            raise ValueError('key of ' + key + ' in expected is not NaN whereas the dataset result is EXPECTED: ' + str(res) + ' and RESULTS: ' + str(data))
        
        diff = abs(res[key] - data[key])
        if diff > MAX_DIFF:
            raise ValueError('key of ' + key + ' saw a difference over ' + str(MAX_DIFF) + ' expected ' + str(res[key]) + ' got ' + str(data[key]) + ' RESULTS: ' + str(data))


class TestStatistics(unittest.TestCase):
    def _compare(self, data, res):
        if not isinstance(data, np.ndarray):
            data = self._data(data)  # convert to a numpy array
        
        pred = compute_statistics(data)
        data = pred.json()  # get the dict

        # run the common compare function for statistics
        compare_statistics(data, res)

    def _data(self, arr):
        return np.array(arr, dtype=np.double)

    def test_random(self):
        self._compare(
            [1,2,3,4,5,6,20,10,5,3,3,3,5,-100,0.25,0.1],
            {
                'sum': -29.65,
                'mean': -1.853125,
                'variance': 663.3204589,
                'STD': 25.75500842524,
                'CV': -13.8991,
                'Q1': 1.5,
                'Q3': 5.0,
                'median': 3.0,
                'IQR': 3.5,
                'min': -100,
                'max': 20,
                'range': 120
            }
        )

    def test_odd(self):
        self._compare(
            [8,4,1,2,5,7,9,-10,20,1312.532,0.123123],
            {
                'sum': 1358.655123,
                'mean': 123.51410209091,
                'variance': 141424.29078316,
                'STD': 376.06421098418,
                'CV': 3.0447,
                'Q1': 1,
                'Q3': 9,
                'median': 5,
                'IQR': 8,
                'min': -10,
                'max': 1312.532,
                'range': 1322.532
            }
        )
    
    def test_none(self):
        self._compare(
            [],
            {
                'sum': NAN,
                'mean': NAN,
                'variance': NAN,
                'STD': NAN,
                'CV': NAN,
                'Q1': NAN,
                'Q3': NAN,
                'median': NAN,
                'IQR': NAN,
                'min': NAN,
                'max': NAN,
                'range': NAN
            }
        )

    def test_small(self):
        self._compare(
            [1],
            {
                'sum': 1,
                'mean': 1,
                'variance': 0,
                'STD': 0,
                'CV': 0,
                'Q1': NAN,
                'Q3': NAN,
                'median': 1,
                'IQR': NAN,
                'min': 1,
                'max': 1,
                'range': 0
            }
        )

    def test_quartile(self):
        self._compare(
            [3, 2, 1],
            {
                'Q1': 1,
                'Q3': 3,
                'median': 2,
                'IQR': 2
            }
        )

        self._compare(
            [3, 1],
            {
                'Q1': 1,
                'Q3': 3,
                'median': 2,
                'IQR': 2
            }
        )

        self._compare(
            [-1, 1],
            {
                'Q1': -1,
                'Q3': 1,
                'median': 0,
                'IQR': 2
            }
        )