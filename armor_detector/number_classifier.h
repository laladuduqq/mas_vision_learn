/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-08-02 08:45:21
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-08-02 08:45:37
 * @FilePath: /learn1/armor_detector/number_classifier.h
 * @Description: 
 */
#ifndef _NUMBER_CLASSIFIER_H_
#define _NUMBER_CLASSIFIER_H_

#include <string>
#include <vector>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "armor_types.h"

namespace armor_detector {

// Class used to classify the number of the armor, based on the MLP model
class NumberClassifier {
public:
    NumberClassifier(const std::string &model_path,
                    const std::string &label_path,
                    const double threshold,
                    const std::vector<std::string> &ignore_classes = {});

    // Extract the roi image of number from the src
    cv::Mat extractNumber(const cv::Mat &src, const Armor &armor) const noexcept;

    // Classify the number of the armor
    void classify(const cv::Mat &src, Armor &armor) noexcept;

    // Erase the ignore classes
    void eraseIgnoreClasses(std::vector<Armor> &armors) noexcept;

    double threshold;

private:
    std::mutex mutex_;
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    std::vector<std::string> ignore_classes_;
};

}  // namespace armor_detector

#endif // _NUMBER_CLASSIFIER_H_