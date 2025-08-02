#ifndef _ARMOR_TYPES_H_
#define _ARMOR_TYPES_H_

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace armor_detector {

// Armor size, Unit: mm (converted from original meters)
constexpr double SMALL_ARMOR_WIDTH = 133.0;  // mm
constexpr double SMALL_ARMOR_HEIGHT = 50.0;  // mm
constexpr double LARGE_ARMOR_WIDTH = 225.0;  // mm
constexpr double LARGE_ARMOR_HEIGHT = 50.0;  // mm

// 15 degree in rad
constexpr double FIFTEEN_DEGREE_RAD = 15 * CV_PI / 180;

// Enemy color enum
enum class EnemyColor {
    RED = 0,
    BLUE = 1,
    WHITE = 2,
    UNKNOWN = 3
};

inline std::string enemyColorToString(EnemyColor color) {
    switch (color) {
        case EnemyColor::RED:
            return "RED";
        case EnemyColor::BLUE:
            return "BLUE";
        case EnemyColor::WHITE:
            return "WHITE";
        default:
            return "UNKNOWN";
    }
}

// Armor type enum
enum class ArmorType {
    SMALL,
    LARGE,
    INVALID
};

inline std::string armorTypeToString(const ArmorType &type) {
    switch (type) {
        case ArmorType::SMALL:
            return "small";
        case ArmorType::LARGE:
            return "large";
        default:
            return "invalid";
    }
}

// Struct used to store the light bar
struct Light : public cv::RotatedRect {
    Light() = default;
    
    explicit Light(const std::vector<cv::Point> &contour)
        : cv::RotatedRect(cv::minAreaRect(contour)), color(EnemyColor::UNKNOWN) {
        
        if (contour.empty()) return;

        // Calculate center point
        center = std::accumulate(
            contour.begin(),
            contour.end(),
            cv::Point2f(0, 0),
            [n = static_cast<float>(contour.size())](const cv::Point2f &a, const cv::Point &b) {
                return a + cv::Point2f(b.x, b.y) / n;
            });

        // Get the four corners of the rotated rectangle
        cv::Point2f p[4];
        this->points(p);
        
        // Sort points by y-coordinate to find top and bottom
        std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        // Calculate length and width
        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        // Calculate axis vector (normalized)
        axis = top - bottom;
        if (cv::norm(axis) > 0) {
            axis = axis / cv::norm(axis);
        }

        // Calculate the tilt angle
        // The angle is the angle between the light bar and the horizontal line
        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    EnemyColor color;
    cv::Point2f top, bottom, center;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;
};

// Struct used to store the armor
struct Armor {
    static constexpr const int N_LANDMARKS = 6;
    static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;
    
    Armor() = default;
    
    Armor(const Light &l1, const Light &l2) {
        if (l1.center.x < l2.center.x) {
            left_light = l1;
            right_light = l2;
        } else {
            left_light = l2;
            right_light = l1;
        }

        center = (left_light.center + right_light.center) / 2;
    }

    // Build the points in the object coordinate system, start from bottom left in
    // clockwise order
    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double &w,
                                                         const double &h) noexcept {
        if constexpr (N_LANDMARKS == 4) {
            return {PointType(0, w / 2, -h / 2),
                    PointType(0, w / 2, h / 2),
                    PointType(0, -w / 2, h / 2),
                    PointType(0, -w / 2, -h / 2)};
        } else {
            return {PointType(0, w / 2, -h / 2),
                    PointType(0, w / 2, 0),
                    PointType(0, w / 2, h / 2),
                    PointType(0, -w / 2, h / 2),
                    PointType(0, -w / 2, 0),
                    PointType(0, -w / 2, -h / 2)};
        }
    }

    // Landmarks start from bottom left in clockwise order
    std::vector<cv::Point2f> landmarks() const {
        if constexpr (N_LANDMARKS == 4) {
            return {left_light.bottom, left_light.top, right_light.top, right_light.bottom};
        } else {
            return {left_light.bottom,
                    left_light.center,
                    left_light.top,
                    right_light.top,
                    right_light.center,
                    right_light.bottom};
        }
    }

    // Light pairs part
    Light left_light, right_light;
    cv::Point2f center;
    ArmorType type;

    // Number part
    cv::Mat number_img;
    std::string number;
    float confidence;
    std::string classification_result;
};

// Debug structures
struct DebugLight {
    float center_x;
    float ratio;
    float angle;
    bool is_light;
};

struct DebugArmor {
    std::string type;
    float center_x;
    float light_ratio;
    float center_distance;
    float angle;
};

}  // namespace armor_detector

#endif // _ARMOR_TYPES_H_
