/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-07-28 18:10:53
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-08-02 17:47:25
 * @FilePath: /learn1/thread/armor_detector_thread.cpp
 * @Description: 
 */
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

#include "armor_detector.h"

extern std::atomic<bool> running;
extern std::queue<cv::Mat> frameQueue;
extern std::mutex queueMutex;
extern std::condition_variable queueCond;
extern std::atomic<bool> camReady;


void processThreadFunc() {
    while (!camReady && running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
        //默认参数
        int binary_thres = 90;
        armor_detector::EnemyColor detect_color = armor_detector::EnemyColor::RED;
        bool show_debug = true;
        armor_detector::Detector::LightParams light_params;
        light_params.min_ratio = 0.1;
        light_params.max_ratio = 0.4;
        light_params.max_angle = 40.0;
        light_params.color_diff_thresh = 50;
        armor_detector::Detector::ArmorParams armor_params;
        armor_params.min_light_ratio = 0.7;
        armor_params.min_small_center_distance = 1.0;
        armor_params.max_small_center_distance = 3.2;
        armor_params.min_large_center_distance = 3.2;
        armor_params.max_large_center_distance = 5.5;
        armor_params.max_angle = 35.0;

        // 读取 JSON 配置
        std::string config_path = "config/armor_detector.json";
        try {
            cv::FileStorage fs(config_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            if (fs.isOpened()) {
                auto node = fs["armor_detector"];
                if (!node.empty()) {
                    if (node["debug"].isInt() || node["debug"].isReal())
                        show_debug = (bool)(int)node["debug"];
                    if (node["binary_thres"].isInt())
                        binary_thres = (int)node["binary_thres"];
                    if (node["detect_color"].isString()) {
                        std::string color_str = (std::string)node["detect_color"];
                        detect_color = (color_str == "RED") ? armor_detector::EnemyColor::RED : armor_detector::EnemyColor::BLUE;
                    }
                    auto light_node = node["light_params"];
                    if (!light_node.empty()) {
                        light_params.min_ratio = (double)light_node["min_ratio"];
                        light_params.max_ratio = (double)light_node["max_ratio"];
                        light_params.max_angle = (double)light_node["max_angle"];
                        light_params.color_diff_thresh = (int)light_node["color_diff_thresh"];
                    }
                    auto armor_node = node["armor_params"];
                    if (!armor_node.empty()) {
                        armor_params.min_light_ratio = (double)armor_node["min_light_ratio"];
                        armor_params.min_small_center_distance = (double)armor_node["min_small_center_distance"];
                        armor_params.max_small_center_distance = (double)armor_node["max_small_center_distance"];
                        armor_params.min_large_center_distance = (double)armor_node["min_large_center_distance"];
                        armor_params.max_large_center_distance = (double)armor_node["max_large_center_distance"];
                        armor_params.max_angle = (double)armor_node["max_angle"];
                    }
                }
                fs.release();
            } else {
                std::cerr << "无法打开配置文件，使用默认参数" << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "配置文件读取异常: " << e.what() << std::endl;
        }

        // 用配置初始化Detector
        armor_detector::Detector detector(binary_thres, detect_color, light_params, armor_params);

        // Initialize number classifier (optional)
        try {
            std::string model_path = "/home/pan/code/opencv/learn1/armor_detector/model/lenet.onnx";
            std::string label_path = "/home/pan/code/opencv/learn1/armor_detector/model/label.txt";
            
            detector.classifier = std::make_unique<armor_detector::NumberClassifier>(
                model_path, label_path, 0.7, std::vector<std::string>{"negative"});
            
            std::cout << "Number classifier initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to initialize number classifier: " << e.what() << std::endl;
            std::cerr << "Continuing without number classification..." << std::endl;
        }

        auto last_time = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        double fps = 0.0;
    
    while (running) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCond.wait(lock, [] { return !frameQueue.empty() || !running; });
        if (!frameQueue.empty()) {
            cv::Mat frame = std::move(frameQueue.front());
            frameQueue.pop();
            lock.unlock();
            
            // Detect armors
            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<armor_detector::Armor> armors = detector.detect(frame);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Calculate detection time
            auto detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            // Calculate FPS
            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
            if (elapsed >= 1) {
                fps = frame_count / static_cast<double>(elapsed);
                frame_count = 0;
                last_time = current_time;
            }

            // Draw results
            {
                // 1. 获取三幅图像
                cv::Mat result_img = frame.clone();
                detector.drawResults(result_img);
                cv::Mat binary = detector.getBinaryImage();
                cv::Mat numbers = detector.getAllNumbersImage();

                // 2. 绘制性能和装甲板信息
                std::string fps_text = cv::format("FPS: %.1f", fps);
                std::string detection_text = cv::format("Detection: %ld ms", detection_time);
                std::string armor_count_text = cv::format("Armors: %zu", armors.size());
                cv::putText(result_img, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
                cv::putText(result_img, detection_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
                cv::putText(result_img, armor_count_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

                // 绘制每个装甲板类型和数字
                int y0 = 120;
                for (size_t i = 0; i < armors.size(); ++i) {
                    std::string armor_info = cv::format("%s (%s)", 
                        armor_detector::armorTypeToString(armors[i].type).c_str(),
                        armors[i].number.c_str());
                    cv::putText(result_img, armor_info, cv::Point(10, y0 + 30 * i), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                }

                // 3. 统一高度，自适应缩放
                int show_height = 640;
                auto resize_keep_aspect = [show_height](const cv::Mat& src) {
                    if (src.empty()) return cv::Mat(show_height, show_height, CV_8UC3, cv::Scalar(0,0,0));
                    int w = src.cols * show_height / src.rows;
                    cv::Mat dst;
                    if (src.channels() == 1)
                        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
                    else
                        dst = src;
                    cv::resize(dst, dst, cv::Size(w, show_height));
                    return dst;
                };

                cv::Mat show_result = resize_keep_aspect(result_img);
                cv::Mat show_binary = resize_keep_aspect(binary);
                cv::Mat show_numbers = resize_keep_aspect(numbers);

                // 4. 横向拼接
                std::vector<cv::Mat> imgs = {show_result, show_binary, show_numbers};
                cv::Mat concat_img;
                cv::hconcat(imgs, concat_img);

                // 5. 显示
                cv::imshow("armor_detector_debug", concat_img);
            }
            // Print armor information to console
            if (!armors.empty()) {
                std::cout << "\rDetected " << armors.size() << " armors: ";
                for (const auto& armor : armors) {
                    std::cout << armor_detector::armorTypeToString(armor.type) << "(" << armor.number << ") ";
                }
                std::cout << std::flush;
            }

            int key = cv::waitKey(1);
            if (key == 27) running = false;
        }
    }
}