#include "Join.h"

using namespace fl;
using std::vector;

namespace util {

Tensor join1DTensors(const vector<Tensor>& tensors) {
  int totalSize = 0;
  for (auto t: tensors)
    totalSize += t.dim(0);
  Tensor out({totalSize});//, arrays[0].type());
  int i = 0;
  for (auto t: tensors) {
    const auto l = t.dim(0);
    out(range(i, i+l)) = t;
    i += l;
  }
  return out;
}


// stack on dim 1
Tensor stack1DTensors(const vector<Tensor>& tensors) {
  const auto& examplar = tensors.front();
  const int m = examplar.shape().dim(0);
  const int n = tensors.size();
  Tensor out({m, n});
  int i = 0;
  for (auto t: tensors) {
    out(fl::span, i) = t;
    i += 1;
  }
  return out;
}

}
