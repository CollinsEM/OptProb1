#pragma once

// Asserts are used for testing and debugging purposes only.

// When running performance tests, uncomment this line or pass
// -DNDEBUG to the compiler.
// #define NDEBUG
#include <cassert>
#include <cstring>
#include <chrono>
#include <random>

#include <emc_utils.h>

// Simple array data structure.
template<typename T>
struct Vec {
  size_t vecSize;
  T *data;
  // Create a test vector with sz elements, mean value of mu, and
  // standard deviation of sigma.
  Vec(const size_t sz) : vecSize(sz), data(NULL) {
    data = new T[sz];
    memset(data, 0, sz*sizeof(T));
  }
  ~Vec() {
    if (data) delete [] data;
  }
  T & operator[](const size_t idx) {
    assert(idx<vecSize);
    return data[idx];
  }
  const T & operator[](const size_t idx) const {
    assert(idx<vecSize);
    return data[idx];
  }
};

template<typename T>
struct TestVec {
  size_t vecSize;
  T *data;
  std::normal_distribution<T> dist;
  static std::default_random_engine randGen;
  // Create a test vector with sz elements, mean value of mu, and
  // standard deviation of sigma.
  TestVec(const size_t sz, T mu, T sigma)
    : vecSize(sz), data(NULL), dist(mu, sigma) {
    data = new T[sz];
    memset(data, 0, sz*sizeof(T));
  }
  ~TestVec() {
    if (data) delete [] data;
  }
  void init() {
    for (size_t i=0; i<vecSize; ++i) {
      data[i] = dist(randGen);
    }
  }
  T & operator[](const size_t idx) {
    assert(idx<vecSize);
    return data[idx];
  }
  const T & operator[](const size_t idx) const {
    assert(idx<vecSize);
    return data[idx];
  }
};

// Initialize the random number generator used in the TestVec class
template<typename T>
std::default_random_engine TestVec<T>::randGen(std::chrono::system_clock::now().time_since_epoch().count());

// Compute the output vector from the input vectors. This will serve
// as the reference solution. We will not bother to optimize this
// version as it will serve only to verify our optimized solutions.
template<typename T>
struct SolnVec {
  size_t vecSize;
  T *data;
  // Internal data for storing intermediate values needed to compute
  // the output vector.
  Vec<T> X;
  OnlineAverage<T> xStats;

public:
  // Create a test vector with sz elements, mean value of mu, and
  // standard deviation of sigma.
  SolnVec(const size_t sz)
    : vecSize(sz), data(NULL), X(sz) {
    data = new T[sz];
    memset(data, 0, sz*sizeof(T));
  }
  ~SolnVec() {
    if (data) delete [] data;
  }
  void eval(const TestVec<T> & x, const TestVec<T> & v,
            const TestVec<T> & b, const TestVec<T> & gamma,
            const TestVec<T> & beta) {
    xStats.reset();
    for (size_t i=0; i<vecSize; ++i) {
      X[i] = x[i] + v[i] + b[i];
      xStats += X[i];
    }
    const T xBar = xStats.mean();
    const T xVar = xStats.variance();
    const T rVar = 1.0/(sqrt(xVar)+1e-8);
    for (size_t i=0; i<vecSize; ++i) {
      data[i] = (X[i] - xBar)*rVar*gamma[i] + beta[i];
    }
  }
  T & operator[](const size_t idx) {
    assert(idx<vecSize);
    return data[idx];
  }
  const T & operator[](const size_t idx) const {
    assert(idx<vecSize);
    return data[idx];
  }
};
