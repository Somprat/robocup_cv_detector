#pragma once

#include <array>
#include <string>

namespace booster_vision {

struct CameraIntrin {
    double fx = 643.898;
    double fy = 643.216;
    double cx = 649.038;
    double cy = 357.21;
    std::array<double, 5> distortion_coeffs = {-0.0553056, 0.065975, -0.000994232, 2.98548e-05,
                                               -0.0216579};
    int distortion_model = 2;  // 0: none, 1: opencv, 2: rs d455 only
};

struct CameraExtrin {
    std::array<std::array<double, 4>, 4> matrix = {
            {{0.05659255, 0.03014561, 0.99794215, 0.04217854},
             {-0.99839673, 0.00283376, 0.05653273, 0.01405556},
             {-0.00112372, -0.9995415, 0.03025765, -0.01538294},
             {0.0, 0.0, 0.0, 1.0}}};
};

struct CameraConfig {
    std::string type = "realsense";  // or "zed"
    CameraIntrin intrin;
    CameraExtrin extrin;
    double pitch_compensation = 0.0;
    double yaw_compensation = 0.0;
    double z_compensation = 0.0;
};

enum class DetectorBackend {
    OpenCvDnn,
    TensorRT,
};

enum class ModelProfile {
    NUbots,
    Booster,
};

// instead of getting config from the engine model, we specify it manually instead
struct DetectionModelConfig {
    DetectorBackend backend = DetectorBackend::OpenCvDnn;
    // Selects preprocessing and class interpretation, independently of the backend.
    ModelProfile profile = ModelProfile::NUbots;
    // Required: the caller supplies the model file path before detector creation.
    std::string model_path;
    double confidence_threshold = 0.2;
    // Intersection-over-union threshold for suppressing duplicate boxes.
    double nms_threshold = 0.5;
    // Input dimensions are read from the model and validated by its profile.
};

struct BallPoseEstimatorConfig {
    bool use_depth = false;
    double radius = 1.109;
    double down_sample_leaf_size = 0.01;
    double cluster_distance_threshold = 0.01;
    double fitting_distance_threshold = 0.01;
};

struct HumanLikePoseEstimatorConfig {
    bool use_depth = false;
    double down_sample_leaf_size = 0.01;
    double fitting_distance_threshold = 0.01;
    double statistic_outlier_multiplier = 0.01;
};

struct VisionConfig {
    bool show_res = false;
    CameraConfig camera;
    DetectionModelConfig detection_model;
    bool use_depth = true;
    BallPoseEstimatorConfig ball_pose_estimator;
    HumanLikePoseEstimatorConfig human_like_pose_estimator;
};

inline VisionConfig getDefaultConfig() {
    return VisionConfig{};
}

}  // namespace booster_vision
