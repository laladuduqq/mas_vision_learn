/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-07-28 21:37:11
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-08-02 17:31:39
 * @FilePath: /learn1/armor_detector/armor_detector.h
 * @Description: 
 */
#ifndef _ARMOR_DETECTOR_H_
#define _ARMOR_DETECTOR_H_

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "armor_types.h"
#include "number_classifier.h"

namespace armor_detector {

class Detector {
public:
    struct LightParams {
        // width / height
        double min_ratio;
        double max_ratio;
        // vertical angle
        double max_angle;
        // judge color
        int color_diff_thresh;
    };

    struct ArmorParams {
        double min_light_ratio;
        // light pairs distance
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        // horizontal angle
        double max_angle;
    };

    Detector(const int &bin_thres, const EnemyColor &color, const LightParams &l,
             const ArmorParams &a);

    std::vector<Armor> detect(const cv::Mat &input) noexcept;

    cv::Mat preprocessImage(const cv::Mat &input) noexcept;
    std::vector<Light> findLights(const cv::Mat &rgb_img,
                                  const cv::Mat &binary_img) noexcept;
    std::vector<Armor> matchLights(const std::vector<Light> &lights) noexcept;

    // For debug usage
    cv::Mat getAllNumbersImage() const noexcept;
    void drawResults(cv::Mat &img) const noexcept;

    // Parameters
    int binary_thres;
    EnemyColor detect_color;
    LightParams light_params;
    ArmorParams armor_params;

    std::unique_ptr<NumberClassifier> classifier;

    // Debug data
    cv::Mat binary_img;
    std::vector<DebugLight> debug_lights;
    std::vector<DebugArmor> debug_armors;

    // Debug接口
    const cv::Mat& getBinaryImage() const noexcept { return binary_img; }
    const std::vector<DebugLight>& getDebugLights() const noexcept { return debug_lights; }
    const std::vector<DebugArmor>& getDebugArmors() const noexcept { return debug_armors; }

private:
    bool isLight(const Light &possible_light) noexcept;
    bool containLight(const int i, const int j, const std::vector<Light> &lights) noexcept;
    ArmorType isArmor(const Light &light_1, const Light &light_2) noexcept;

    cv::Mat gray_img_;

    std::vector<Light> lights_;
    std::vector<Armor> armors_;
};

} // namespace armor_detector

#endif // _ARMOR_DETECTOR_H_