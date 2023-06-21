#include "ChooseAlong.h"

using namespace fl;

namespace util {

Tensor flattenForChooseAlongDim1(Tensor i, int d0) {
  const int  n = i.shape().dim(0);
  return d0 * arange(0, n, 1, i.type()) + i;
}


Tensor chooseAlongDim1(Tensor v, Tensor i) {
  const auto flattened = v.flatten();
  const auto fi = flattenForChooseAlongDim1(i, v.shape().dim(0));
  return flattened(fi);
}


Variable chooseAlongDim1(Variable v, Tensor i) {
  const auto flattened = flat(v);
  const auto fi = flattenForChooseAlongDim1(i, v.shape().dim(0));
  return flattened(fi);
}

} // namespace util
