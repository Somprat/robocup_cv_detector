#include "impl.h"

#include <array>
#include <stdexcept>

#include "../detector.h"

namespace booster_vision {
namespace onnx {
namespace {

    // A file for mapping NUbot onnx classes to HTWK classes

// NUbots: ball, goalpost, robot, L-, T-, X-Intersection.
// The generic robot class maps to Opponent by integration policy; the model
// itself does not distinguish opponents from teammates.
constexpr std::array<int, 6> kNUbotsToHtwk{{0, 1, 7, 3, 4, 5}};

}  // namespace

DetectionRes MakeDetection(ModelProfile profile, int model_class_id,
                           const cv::Rect& bbox, float confidence) {
    int htwk_class_id;
    switch (profile) {
        case ModelProfile::NUbots:
            if (model_class_id < 0 ||
                model_class_id >= static_cast<int>(kNUbotsToHtwk.size())) {
                throw std::out_of_range("Invalid NUbots model class ID: " + std::to_string(model_class_id));
            }
            htwk_class_id = kNUbotsToHtwk[model_class_id];
            break;
        case ModelProfile::Booster:
            if (model_class_id < 0 ||
                model_class_id >= static_cast<int>(YoloV8Detector::kClassLabels.size())) {
                throw std::out_of_range("Invalid Booster model class ID: " + std::to_string(model_class_id));
            }
            htwk_class_id = model_class_id;
            break;
        default:
            throw std::invalid_argument("Unknown detector model profile");
    }

    return DetectionRes{bbox, htwk_class_id,
                        YoloV8Detector::kClassLabels.at(htwk_class_id), confidence};
}

}  // namespace onnx
}  // namespace booster_vision
