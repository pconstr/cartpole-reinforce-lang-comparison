#pragma once

#include <boost/circular_buffer.hpp>

namespace util {

using boost::circular_buffer;

template<class T>
class RollingMean {
 public:
  RollingMean(int size) :
    elements(size),
    current_mean(0.0)
  {
  }

  T mean() const {
    return current_mean;
  }

  void add(T v) {
    if (elements.full()) {
      const auto leaving = *elements.begin();
      current_mean +=  (v - leaving) / elements.size();
    } else if (elements.empty()) {
      current_mean = v;
    } else {
      current_mean += (v - current_mean) / (elements.size() + 1);
    }
    elements.push_back(v);
  }

private:
  float current_mean;
  circular_buffer<T> elements;
};

} // namespace util
