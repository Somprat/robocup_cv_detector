#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/data_types.h"
#include "config.h"

namespace booster_vision {

class YoloV8Detector {
public:
    virtual ~YoloV8Detector() = default;
    virtual void Init(std::string model_path){};
    virtual std::vector<DetectionRes> Inference(const cv::Mat &img) = 0;
    virtual Detections proceed(uint8_t* img_raw) = 0;
    void setConfidenceThreshold(float confidence_threshold) {
        confidence_ = confidence_threshold;
    }

    float getConfidenceThreshold() {
        return confidence_;
    }

    std::string getModelPath() {
        return model_path_;
    }

    static std::shared_ptr<YoloV8Detector> CreateYoloV8Detector(
        const DetectionModelConfig& config);
    static const std::vector<std::string> kClassLabels;

protected:
    YoloV8Detector(const std::string &model_path, float conf, float nms_threshold) :
        model_path_(model_path), confidence_(conf), nms_threshold_(nms_threshold) {
    }

    std::string model_path_;
    float confidence_;
    float nms_threshold_;
};

} // namespace booster_vision
