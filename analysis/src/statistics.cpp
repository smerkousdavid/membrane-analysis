#include <math.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include "../include/statistics.hpp"

namespace statistics {
    template<typename Iter>
    const double get_sum(Iter begin, Iter end) {
        // there's no count
        if (begin == end) {
            return std::nan("0");
        }

        double sum = 0;
        while (begin != end) {
            sum += static_cast<double>(*begin);

            // shift to next
            begin++;
        }
        return sum;
    }

    template<typename Iter>
    const double get_mean(Iter begin, Iter end) {
        double sum = 0;
        uint32_t count = 0U;
        while (begin != end) {
            sum += static_cast<double>(*begin);

            // shift to next
            begin++;
            count++;
        }

        // there's no count
        if (count == 0U) {
            return std::nan("0/0");
        }

        return sum / static_cast<double>(count);
    }

    const std::pair<bool, uint32_t> get_median_index(const uint32_t start, const uint32_t end) {
        // given a size of an array return the median index (true if even dataset)
        uint32_t n = end - start; // get total count
        if (n % 2 == 0) {  // operate on even
            n = (n / 2) - 1; // offset to index
            return std::pair<bool, uint32_t>(true, n + start);
        }
        // operate on odd
        n = (n - 1) / 2;
        return std::pair<bool, uint32_t>(false, n + start);
    }

    const std::tuple<bool, uint32_t, double> get_median(const std::vector<double> &data, const uint32_t size, const uint32_t start, const uint32_t end) {
        // in the case where there's no data to get a median on or if the range specified isn't valid
        if (start >= end || size < 1) {
            double med = std::nan("0");
            const uint32_t ind = (start > size) ? (size - 1) : start;
            if (size > 0 && ind < size) {
                med = data.at(ind);
            }
            return std::tuple<bool, uint32_t, double>(false, ind, med); // return the 0th in most cases
        }
        
        // let's get the index and determine even-ness
        const std::pair<bool, uint32_t> med = get_median_index(start, end);
        const bool is_even = med.first;

        // get average if even dataset
        double data_median;
        if (is_even && med.second + 1 < size && size >= 2) { // for median of even to work we need a size of 2
            data_median = (data.at(med.second) + data.at(med.second + 1)) / 2.0;
        } else if (med.second < size) {
            data_median = data.at(med.second); // odd dataset so it's the center value
        } else {
            data_median = std::nan("0/0");
        }

        return std::tuple<bool, uint32_t, double>(is_even, med.second, data_median);
    }

    template<typename T>
    const std::tuple<double, double, double, double, double, double, double> get_quartiles(std::vector<T> &org_data) {
        /* returns form of (min, Q1, median/Q2, Q3, max, range) */
        const uint32_t size = static_cast<uint32_t>(org_data.size());
        if (size == 0U) {
            // invalid quartile range
            double dummy = std::nan("0");
            return std::tuple<double, double, double, double, double, double, double>(dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        }

        // sort the data (make a copy) as the order might matter for original
        std::vector<double> data;
        std::vector<T>::iterator d_it = org_data.begin();
        while (d_it != org_data.end()) {
            data.push_back(static_cast<double>(*d_it));
            d_it++;
        }
        std::sort(data.begin(), data.end());

        // get overal median
        std::tuple<bool, uint32_t, double> median = get_median(data, size, 0U, size);

        // quartile 1: make sure in odd dataset we're below limits when adding a value (is_even) ? current index : mext index
        uint32_t new_end = (std::get<0>(median)) ? (std::min(std::get<1>(median) + 1, size - 1)) : (std::get<1>(median));
        std::tuple<bool, uint32_t, double> medianQ1 = get_median(data, size, 0U, new_end);

        // quartile 3: same condition as above
        uint32_t new_start = (std::min(std::get<1>(median) + 1, size - 1));  // (std::get<0>(median)) ? (std::get<1>(median)) : (std::min(std::get<1>(median) + 1, size - 1));
        std::tuple<bool, uint32_t, double> medianQ3 = get_median(data, size, new_start, size);

        // let's make sure we can get Q1 and Q3 from dataset, which is possible on 2 or more points, but not just 1
        if (size < 2) {
            std::get<2>(medianQ1) = std::nan("0");
            std::get<2>(medianQ3) = std::nan("0");
        }

        // get the IQR from Q1 -> Q3
        double IQR;
        if (std::isnan(std::get<2>(medianQ1)) || std::isnan(std::get<2>(medianQ3))) {
            IQR = std::nan("Q1/Q2 nan");
        } else {
            IQR = std::get<2>(medianQ3) - std::get<2>(medianQ1); // compute IQR
        }

        // get the min and max (it's already sorted so it's just the first and last elements)
        double min = data.at(0);
        double max = data.at(size - 1);
        double range = max - min;
        return std::tuple<double, double, double, double, double, double, double>(min, std::get<2>(medianQ1), std::get<2>(median), std::get<2>(medianQ3), IQR, max, range);
    }

    template<typename Iter>
    const double get_variance(const double mean, Iter begin, Iter end) {
        if (std::isnan(mean)) {
            return std::nan("0");
        }

        double temp = 0;
        uint32_t count = 0U;
        while (begin != end) {
            temp += (static_cast<double>(*begin) - mean) * (static_cast<double>(*begin) - mean);
            
            // shift to next
            begin++;
            count++;
        }

        // there's no count
        if (count == 0U) {
            return 0.0;
        }

        return temp / static_cast<double>(count);
    }

    const double get_standard_deviation(const double variance) {
        // there's no variance
        if (std::isnan(variance)) {
            return std::nan("0");
        } else if (variance == 0) {
            return 0.0;
        }

        return std::sqrt(variance);
    }

    const double get_coef_of_variation(const double mean, const double standard_deviation) {
        // variance only works on positive mean values... no descriptors on negative units
        if (std::isnan(mean)) {
            return std::nan("0");
        } else if (mean > -0.001 && mean < 0.001) {
            return std::nan("0/0");
        }

        return standard_deviation / mean;
    }

    StatsResults::StatsResults() {
        this->valid = false;
    }

    const bool StatsResults::is_valid() {
        return this->valid;
    }

    const bool StatsResults::has_quartile() {
        return (!std::isnan(this->min) && !std::isnan(this->max));
    }

    const bool StatsResults::is_nan(const double val) {
        return std::isnan(val); // wrapper makes it easier to expose to python
    }

    template<typename T>
    StatsResults compute_stats(std::vector<T> &data) {
        // construct the results
        StatsResults stats;
        
        // make sure it's valid
        if (data.empty()) {
            stats.valid = false;
            const double nan = std::nan("0");
            stats.sum = nan;
            stats.mean = nan;
            stats.variance = nan;
            stats.STD = nan;
            stats.coeff_variation = nan;
            stats.min = nan;
            stats.Q1 = nan;
            stats.median = nan;
            stats.Q3 = nan;
            stats.IQR = nan;
            stats.max = nan;
            stats.range = nan;
            return stats;
        }

        // basic statistics
        stats.valid = true;
        stats.sum = get_sum(data.begin(), data.end());
        if (std::isnan(stats.sum)) {
            stats.mean = std::nan("0/0");
        } else {
            stats.mean = stats.sum / static_cast<double>(data.size());  // divide by the total size
        }
        stats.variance = get_variance(stats.mean, data.begin(), data.end());
        stats.STD = get_standard_deviation(stats.variance);
        stats.coeff_variation = get_coef_of_variation(stats.mean, stats.STD);

        // quartiles
        // min, medianQ1[2], median[2], medianQ3[2], IQR, max, range
        std::tie(stats.min, stats.Q1, stats.median, stats.Q3, stats.IQR, stats.max, stats.range) = get_quartiles(data);
        return stats;
    }
}