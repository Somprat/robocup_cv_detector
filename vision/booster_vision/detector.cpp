#include "detector.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#if defined(TRT_6_0)
#include "trt_6.0/impl.h"
#elif defined(TRT_6_2)
#include "trt_6.2/impl.h"
#endif

#include "config.h"

namespace booster_vision {

const std::vector<std::string> YoloV8Detector::kClassLabels{"Ball", "Goalpost", "Person", "LCross",
                                                            "TCross", "XCross", "PenaltyPoint", "Opponent", "BRMarker"};

std::shared_ptr<YoloV8Detector> YoloV8Detector::CreateYoloV8Detector(
    const DetectionModelConfig& config) {
    try {
        if (config.model_path.empty()) {
            throw std::invalid_argument("Detector model_path must not be empty");
        }
        std::error_code path_error;
        if (!std::filesystem::is_regular_file(config.model_path, path_error)) {
            throw std::invalid_argument(
                "Detector model_path must refer to an existing regular file: " + config.model_path);
        }
        if (!std::ifstream(config.model_path, std::ios::binary).is_open()) {
            throw std::invalid_argument("Detector model file is not readable: " + config.model_path);
        }
        if (!std::isfinite(config.confidence_threshold) ||
            config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0) {
            throw std::invalid_argument("Detector confidence_threshold must be finite and in [0, 1]");
        }
        if (!std::isfinite(config.nms_threshold) ||
            config.nms_threshold < 0.0 || config.nms_threshold > 1.0) {
            throw std::invalid_argument("Detector nms_threshold must be finite and in [0, 1]");
        }
        switch (config.backend) {
            case DetectorBackend::TensorRT:
#if defined(TRT_6_0) || defined(TRT_6_2)
                return std::make_shared<YoloV8DetectorTRT>(
                    config.model_path, static_cast<float>(config.confidence_threshold),
                    static_cast<float>(config.nms_threshold));
#else
                throw std::runtime_error("TensorRT detector backend is unavailable in this build");
#endif
            case DetectorBackend::OpenCvDnn:
                throw std::runtime_error("OpenCV DNN detector backend is not implemented yet");
            default:
                throw std::invalid_argument("Unknown detector backend");
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return nullptr;
    }
}


}
