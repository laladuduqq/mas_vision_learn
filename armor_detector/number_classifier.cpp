/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-08-02 08:45:31
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-08-02 08:50:25
 * @FilePath: /learn1/armor_detector/number_classifier.cpp
 * @Description: 
 */
#include "number_classifier.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <fmt/core.h>
#include <fmt/format.h>

namespace armor_detector {

NumberClassifier::NumberClassifier(const std::string &model_path,
                                   const std::string &label_path,
                                   const double thre,
                                   const std::vector<std::string> &ignore_classes)
    : threshold(thre), ignore_classes_(ignore_classes) {
    
    // Load the ONNX model
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model from: " + model_path);
    }
    
    // Load class labels
    std::ifstream label_file(label_path);
    if (!label_file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + label_path);
    }
    
    std::string line;
    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }
    
    if (class_names_.empty()) {
        throw std::runtime_error("No class names found in label file: " + label_path);
    }
}

cv::Mat NumberClassifier::extractNumber(const cv::Mat &src, const Armor &armor) const noexcept {
    // Light length in image
    static const int light_length = 12;
    // Image size after warp
    static const int warp_height = 28;
    static const int small_armor_width = 32;
    static const int large_armor_width = 54;
    // Number ROI size
    static const cv::Size roi_size(20, 28);
    static const cv::Size input_size(28, 28);

    // Warp perspective transform
    cv::Point2f lights_vertices[4] = {
        armor.left_light.bottom, armor.left_light.top, 
        armor.right_light.top, armor.right_light.bottom};

    const int top_light_y = (warp_height - light_length) / 2 - 1;
    const int bottom_light_y = top_light_y + light_length;
    const int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;
    
    cv::Point2f target_vertices[4] = {
        cv::Point(0, bottom_light_y),
        cv::Point(0, top_light_y),
        cv::Point(warp_width - 1, top_light_y),
        cv::Point(warp_width - 1, bottom_light_y),
    };
    
    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // Get ROI
    number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // Binarize
    cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::resize(number_image, number_image, input_size);
    
    return number_image;
}

void NumberClassifier::classify(const cv::Mat &src, Armor &armor) noexcept {
    // Check if number image is empty
    if (armor.number_img.empty()) {
        armor.confidence = 0.0f;
        armor.number = "unknown";
        armor.classification_result = "unknown:0.0%";
        return;
    }

    // Normalize
    cv::Mat input = armor.number_img / 255.0;

    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob);

    // Set the input blob for the neural network
    std::lock_guard<std::mutex> lock(mutex_);
    net_.setInput(blob);

    // Forward pass the image blob through the model
    cv::Mat outputs = net_.forward().clone();

    // Decode the output
    double confidence;
    cv::Point class_id_point;
    minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;

    // Ensure label_id is within bounds
    if (label_id >= 0 && label_id < static_cast<int>(class_names_.size())) {
        armor.confidence = confidence;
        armor.number = class_names_[label_id];
    } else {
        armor.confidence = 0.0f;
        armor.number = "unknown";
    }

    armor.classification_result = fmt::format("{}:{:.1f}%", armor.number, armor.confidence * 100.0);
}

void NumberClassifier::eraseIgnoreClasses(std::vector<Armor> &armors) noexcept {
    armors.erase(
        std::remove_if(armors.begin(),
                       armors.end(),
                       [this](const Armor &armor) {
                           // Check confidence threshold
                           if (armor.confidence < threshold) {
                               return true;
                           }

                           // Check ignore classes
                           for (const auto &ignore_class : ignore_classes_) {
                               if (armor.number == ignore_class) {
                                   return true;
                               }
                           }

                           // Check armor type mismatch
                           bool mismatch_armor_type = false;
                           if (armor.type == ArmorType::LARGE) {
                               mismatch_armor_type = armor.number == "outpost" || armor.number == "2" ||
                                                     armor.number == "sentry";
                           } else if (armor.type == ArmorType::SMALL) {
                               mismatch_armor_type = armor.number == "1" || armor.number == "base";
                           }
                           
                           return mismatch_armor_type;
                       }),
        armors.end());
}

}  // namespace armor_detector