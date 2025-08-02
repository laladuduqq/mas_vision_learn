/*
 * @Author: laladuduqq 2807523947@qq.com
 * @Date: 2025-07-28 18:10:53
 * @LastEditors: laladuduqq 2807523947@qq.com
 * @LastEditTime: 2025-08-02 16:56:54
 * @FilePath: /learn1/thread/cam_thread.cpp
 * @Description: 
 */
#include "HikCamera.h"
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

extern std::atomic<bool> running;
extern std::queue<cv::Mat> frameQueue;
extern std::mutex queueMutex;
extern std::condition_variable queueCond;
extern std::atomic<bool> camReady;

void cameraThreadFunc(HikCamera& cam) {
    cv::Mat frame;
    if (!cam.openCamera()) {
        printf("打开摄像头失败！\n");
        running = false;
        return;
    }
    camReady = true;
    while (running) {
        if (cam.grabImage(frame)) {
            std::lock_guard<std::mutex> lock(queueMutex);
            // 只保留最新一帧
            while (!frameQueue.empty()) frameQueue.pop();
            frameQueue.push(frame);
            queueCond.notify_one();
        }
    }
    cam.closeCamera();
}