#pragma once

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"

namespace util {

// along dim 1, picking up from dim0
fl::Tensor indexTensorAlong(fl::Tensor v, fl::Tensor i);

// along dim 1, picking up from dim0
fl::Variable chooseAlongDim1(fl::Variable v, fl::Tensor i);

} // namespace util
