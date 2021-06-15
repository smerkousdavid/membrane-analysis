#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>

#ifndef STATISTICS_H
#define STATISTICS_H

namespace statistics {
    template<typename Iter>
    const double get_sum(Iter begin, Iter end);

    template<typename Iter>
    const double get_mean(Iter begin, Iter end);

    template<typename Iter>
    const double get_variance(const double mean, Iter begin, Iter end);

    const double get_standard_deviation(const double variance);
    const double get_coef_of_variation(const double mean, const double standard_deviation);

    template<typename T>
    const std::tuple<double, double, double, double, double, double, double> get_quartiles(std::vector<T> &org_data);

    class StatsResults {
        public:
            // basic stats
            bool valid;
            double sum;
            double mean;
            double variance;
            double STD;
            double coeff_variation;
            
            // quartile stuff
            double min;
            double max;
            double range;
            double Q1;
            double median;
            double Q3;
            double IQR;

            StatsResults();
            const bool is_valid();
            const bool has_quartile();
            const bool is_nan(const double val);
    };

    template<typename T>
    StatsResults compute_stats(std::vector<T> &data);
}

#endif