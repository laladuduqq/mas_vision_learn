#include "hikcamera/include/HikCamera.h"
#include <opencv2/highgui.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

std::atomic<bool> running(true);
std::atomic<bool> camReady(false);
std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCond;

void cameraThreadFunc(HikCamera& cam);
void processThreadFunc();

int main()
{
    HikCamera cam;
    std::thread camThread(cameraThreadFunc, std::ref(cam));
    std::thread procThread(processThreadFunc);

    camThread.join();
    procThread.join();

    printf("exit.\n");
    return 0;
}