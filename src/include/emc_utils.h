#pragma once

#include <cmath>

//--------------------------------------------------------------------
// Kahan's summation algorithm
//--------------------------------------------------------------------
template<typename T>
struct KahanAccumulator {
  T sum, res;
  KahanAccumulator() : sum(0), res(0) {
  }
  KahanAccumulator(const T & init) : sum(init), res(0) {
  }
  template<typename R>
  KahanAccumulator & operator=(const R & rhs) {
    sum = rhs;
    res = 0;
    return *this;
  }
  template<typename R>
  KahanAccumulator<T> & operator+=(const R & rhs) {
    T y = rhs - res;
    T t = sum + y;
    res = (t - sum) - y;
    sum = t;
    return *this;
  }
  template<typename R>
  KahanAccumulator<T> & operator-=(const R & rhs) {
    T y = -rhs - res;
    T t = sum + y;
    res = (t - sum) - y;
    sum = t;
    return *this;
  }
  template<typename R>
  KahanAccumulator<T> & operator*=(const R & rhs) {
    sum *= rhs;
    res *= rhs;
    return *this;
  }
  template<typename R>
  KahanAccumulator<T> & operator/=(const R & rhs) {
    sum /= rhs;
    res /= rhs;
    return *this;
  }
};

/**
 * \brief Determine the ongoing average and standard deviation as new
 * values are accumulated.
 *
 * This is an online running average calculator.  As values are
 * streamed in, they are incorporated into the running mean value (M).
 * The average is updated using the following recurrance relation:
 *
 * a[k] = a[k-1] + (x[k] - a[k-1])/k
 *
 * Notice that as k gets very large, the update term (x[k] - a[k-1])/k
 * gets very small.  If a[k-1] >> (x[k] - a[k-1])/k, then there is a
 * potential for significant losses in precision due to round-off
 * error.  To counter this, we incorporate a Kahan summation method to
 * recover the dropped digits and save them to a correction term (C)
 * that is incorporated into the next update.
 */
template<typename value_t>
struct OnlineAverage {
  typedef unsigned long ulong;
  typedef value_t sum_t;

  ulong N;
  sum_t M;
  sum_t S;
  value_t minVal;
  value_t maxVal;
private:
  sum_t d;

public:
  /**
   * \brief Default constructor
   *
   * Initializes the state for the online average. However, the state
   * is invalid until the first data point is added.
   */
  OnlineAverage() : N(0), M(0), S(0), minVal(1.1e300), maxVal(-1.1e300) {
  }
  /**
   * \brief Operator overload for aggregating values to the set.
   *
   * \param[in] x New data value.
   */
  template<typename T>
  OnlineAverage & operator+=(const T x) {
    N++;
    d = (x - M);
    M += d/N;
    S += d*(x-M);
    // minVal = ( x < minVal ? x : minVal);
    // maxVal = ( x > maxVal ? x : maxVal);
    return *this;
  }
  /**
   * \brief Reset the internal state.
   *
   * This can be used to restart the averaging process from scratch.
   */
  void reset() {
    N = 0;
    M = 0;
    S = 0;
    minVal =  1.1e300;
    maxVal = -1.1e300;
  }
  /**
   * \brief Number of aggregated data points.
   *
   * \return Number of values that have been aggregated.
   */
  ulong count() const {
    return N;
  }
  /**
   * \brief Arithmetic mean
   *
   * \return Arithmetic mean of the aggregated values.
   */
  sum_t mean() const {
    return M;
  }
  /**
   * \brief Minimum value
   *
   * \return Minimum of the aggregated values.
   */
  value_t min() const {
    return minVal;
  }
  /**
   * \brief Maximum value
   *
   * \return Maximum of the aggregated values.
   */
  value_t max() const {
    return maxVal;
  }
  /**
   * \brief Range of values (max-min)
   *
   * \return Difference between the maximum and minimum values.
   */
  value_t range() const {
    return (maxVal - minVal);
  }
  /**
   * \brief Variance
   *
   * \return Variance of the aggregated values.
   */
  sum_t variance() const {
    return (N>1 ? S/(N-1) : 0.0);
  }
  /**
   * \brief Standard deviation
   *
   * \return Standard deviation of the aggregated values.
   */
  sum_t stdDev() const {
    return std::sqrt(variance());
  }
};
