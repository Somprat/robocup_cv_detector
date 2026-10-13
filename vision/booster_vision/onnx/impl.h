#pragma once

#include "../common/data_types.h"
#include "../config.h"

namespace booster_vision {
namespace onnx {

// Convert a decoded prediction to HTWK categories after NMS on raw model IDs.
// Throws std::invalid_argument for unknown profiles and std::out_of_range for
// class IDs outside the selected profile's vocabulary.
DetectionRes MakeDetection(ModelProfile profile, int model_class_id,
                           const cv::Rect& bbox, float confidence);

}  // namespace onnx
}  // namespace booster_vision
