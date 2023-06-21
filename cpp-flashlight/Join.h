#pragma once

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"

#include <vector>

namespace util {

fl::Tensor join1DTensors(const std::vector<fl::Tensor>& tensors);

// stack on dim 1
 fl::Tensor stack1DTensors(const std::vector<fl::Tensor>& tensors);

} // namespace util
