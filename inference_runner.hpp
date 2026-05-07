/**
 * @file inference_runner.hpp
 * @brief YOLO11 推理执行器模块
 *
 * 本模块封装了不同输入源(图片, 视频, 摄像头, 文件夹)的推理流程,
 * 提供统一的高级接口.
 *
 * 推理流程: 输入源 -> 读取帧 -> Letterbox缩放加归一化加CHW转换
 * -> TensorRT推理 -> 解码加NMS -> 绘制结果 -> 保存/显示
 *
 * 教学要点:
 * 将推理流程封装为类, 隐藏底层细节
 * 不同的输入源共享同一套预处理和推理逻辑
 * 视频/摄像头处理需要循环读取帧
 */

#ifndef INFERENCE_RUNNER_HPP
#define INFERENCE_RUNNER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "config.hpp"
#include "types.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "yolo_trt_engine.hpp"

namespace fs = std::filesystem;

class InferenceRunner {
public:
    explicit InferenceRunner(const std::string& modelPath)
        : engine_(modelPath) {}

    void runImage(const std::string& imgPath) {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            std::cerr << "[Runner] Error: Cannot read image: " << imgPath << std::endl;
            return;
        }

        auto tensor = Preprocessor::imageToTensor(Preprocessor::letterbox(img));
        std::vector<Detection> detections;
        engine_.infer(tensor, detections, img.cols, img.rows);

        std::cout << "[Runner] Detected " << detections.size() << " objects" << std::endl;
        Postprocessor::drawDetections(img, detections);

        std::string filename = imgPath.substr(imgPath.find_last_of("/\\") + 1);
        std::string outputPath = Config::OUTPUT_DIR + "/result_" + filename;
        cv::imwrite(outputPath, img);
        std::cout << "[Runner] Saved: " << outputPath << std::endl;

        cv::imshow("YOLO11 Detection", img);
        cv::waitKey(0);
        cv::destroyWindow("YOLO11 Detection");
    }

    void runVideo(const std::string& videoPath) {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "[Runner] Error: Cannot open video: " << videoPath << std::endl;
            return;
        }

        std::string filename = videoPath.substr(videoPath.find_last_of("/\\") + 1);
        std::string outputPath = Config::OUTPUT_DIR + "/result_" + filename;

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(outputPath, fourcc, cap.get(cv::CAP_PROP_FPS),
            cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                     static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

        int frameCount = 0;
        std::cout << "[Runner] Processing video: " << filename << std::endl;

        processFrames(cap, [&](cv::Mat& frame) {
            writer.write(frame);
            (void)frameCount++;
        });

        std::cout << "\n[Runner] Total frames: " << frameCount << std::endl;
        std::cout << "[Runner] Saved: " << outputPath << std::endl;
        writer.release();
    }

    void runCamera() {
        cv::VideoCapture cap(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            std::cerr << "[Runner] Error: Cannot open camera" << std::endl;
            return;
        }

        std::cout << "[Runner] Camera opened. Press ESC to exit." << std::endl;

        processFrames(cap, [](cv::Mat&) {});

        cap.release();
        cv::destroyAllWindows();
    }

    // 批量处理文件夹中的所有图片
    void runFolder(const std::string& folderPath) {
        std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
        int imageCount = 0;

        std::cout << "[Runner] Scanning folder: " << folderPath << std::endl;

        for (const auto& entry : fs::directory_iterator(folderPath)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                std::cout << "\n[Runner] Processing: " << entry.path().string() << std::endl;
                runImage(entry.path().string());
                imageCount++;
            }
        }

        std::cout << "[Runner] Total images processed: " << imageCount << std::endl;
    }

private:
    template<typename F>
    void processFrames(cv::VideoCapture& cap, F&& onFrame) {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame)) {
            auto tensor = Preprocessor::imageToTensor(Preprocessor::letterbox(frame));
            std::vector<Detection> detections;
            engine_.infer(tensor, detections, frame.cols, frame.rows);
            Postprocessor::drawDetections(frame, detections);

            cv::imshow("YOLO11 Detection", frame);
            onFrame(frame);

            if (cv::waitKey(1) == 27) break;
            frameCount++;
            if (frameCount % 30 == 0) {
                std::cout << "\r[Runner] Frame: " << frameCount << std::flush;
            }
        }
    }

    YoloTrtEngine engine_;
};

#endif // INFERENCE_RUNNER_HPP
