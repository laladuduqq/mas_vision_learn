/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-07-26 21:03:28
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-07-28 11:31:12
 * @FilePath: /learn1/main.cpp
 * @Description: 
 */
#include "hikcamera/include/HikCamera.h"
#include <opencv2/highgui.hpp>

int main()
{
    HikCamera cam;
    if (!cam.openCamera()) {
        printf("打开摄像头失败！\n");
        return -1;
    }

    printf("按esc退出...\n");
    cv::Mat frame;
    static double lastTick = cv::getTickCount();
    static int frameCount = 0;
    while (true) {
        if (!cam.grabImage(frame)) {
            printf("获取图像失败！\n");
            break;
        }
        frameCount++;
        double nowTick = cv::getTickCount();
        double time_ms = (nowTick - lastTick) * 1000 / cv::getTickFrequency();
        double fps = frameCount * 1000.0 / time_ms;
        std::string info = cv::format("FPS: %.2f  Time: %.2f ms", fps, time_ms / frameCount);
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);

        cv::imshow("Camera", frame);
        int key = cv::waitKey(1);
        if (key == 27) break; // 按ESC退出
    }

    cam.closeCamera();
    printf("exit.\n");
    return 0;
}