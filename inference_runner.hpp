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
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "types.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "yolo_trt_engine.hpp"

namespace fs = std::filesystem;

// 推理执行器类. 封装完整的推理流程, 支持多种输入源
class InferenceRunner {
public:
    // 构造函数: 初始化TensorRT引擎
    explicit InferenceRunner(const std::string& modelPath)
        : engine_(modelPath) {}

    // 对单张图片进行推理
    void runImage(const std::string& imgPath) {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            std::cerr << "[Runner] Error: Cannot read image: " << imgPath << std::endl;
            return;
        }

        // 预处理
        cv::Mat processed = Preprocessor::letterbox(img);
        std::vector<float> tensor = Preprocessor::imageToTensor(processed);

        // 推理
        std::vector<Detection> detections;
        engine_.infer(tensor, detections, img.cols, img.rows);

        std::cout << "[Runner] Detected " << detections.size() << " objects" << std::endl;

        // 绘制结果
        Postprocessor::drawDetections(img, detections);

        // 保存结果
        std::string filename = imgPath.substr(imgPath.find_last_of("/\\") + 1);
        std::string outputPath = Config::OUTPUT_DIR + "/result_" + filename;
        cv::imwrite(outputPath, img);
        std::cout << "[Runner] Saved: " << outputPath << std::endl;

        // 显示结果
        cv::imshow("YOLO11 Detection", img);
        cv::waitKey(0);
        cv::destroyWindow("YOLO11 Detection");
    }

    // 对视频文件进行逐帧推理
    void runVideo(const std::string& videoPath) {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "[Runner] Error: Cannot open video: " << videoPath << std::endl;
            return;
        }

        std::string filename = videoPath.substr(videoPath.find_last_of("/\\") + 1);
        std::string outputPath = Config::OUTPUT_DIR + "/result_" + filename;

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        int frameWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight));

        cv::Mat frame;
        int frameCount = 0;

        std::cout << "[Runner] Processing video: " << filename << std::endl;

        while (cap.read(frame)) {
            cv::Mat processed = Preprocessor::letterbox(frame);
            std::vector<float> tensor = Preprocessor::imageToTensor(processed);

            std::vector<Detection> detections;
            engine_.infer(tensor, detections, frame.cols, frame.rows);

            Postprocessor::drawDetections(frame, detections);
            writer.write(frame);
            cv::imshow("YOLO11 Detection", frame);

            if (cv::waitKey(1) == 27) break;

            frameCount++;
            if (frameCount % 30 == 0) {
                std::cout << "\r[Runner] Frame: " << frameCount << std::flush;
            }
        }

        std::cout << "\n[Runner] Total frames: " << frameCount << std::endl;
        std::cout << "[Runner] Saved: " << outputPath << std::endl;

        cap.release();
        writer.release();
        cv::destroyAllWindows();
    }

    // 实时摄像头推理
    void runCamera() {
        cv::VideoCapture cap(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            std::cerr << "[Runner] Error: Cannot open camera" << std::endl;
            return;
        }

        std::cout << "[Runner] Camera opened. Press ESC to exit." << std::endl;

        cv::Mat frame;
        int frameCount = 0;

        while (cap.read(frame)) {
            cv::Mat processed = Preprocessor::letterbox(frame);
            std::vector<float> tensor = Preprocessor::imageToTensor(processed);

            std::vector<Detection> detections;
            engine_.infer(tensor, detections, frame.cols, frame.rows);

            Postprocessor::drawDetections(frame, detections);
            cv::imshow("YOLO11 Detection (Press ESC to exit)", frame);

            if (cv::waitKey(1) == 27) break;
            frameCount++;
        }

        std::cout << "[Runner] Total frames processed: " << frameCount << std::endl;
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
    YoloTrtEngine engine_;  // TensorRT推理引擎实例
};

#endif // INFERENCE_RUNNER_HPP
