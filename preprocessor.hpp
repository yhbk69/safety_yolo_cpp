/**
 * @file preprocessor.hpp
 * @brief YOLO11 图像预处理模块
 *
 * 本模块负责将原始图像转换为TensorRT模型所需的输入格式.
 * 预处理流程包含三个关键步骤:
 * 1. Letterbox缩放: 保持宽高比的前提下将图像缩放到模型输入尺寸
 * 2. 颜色空间转换: BGR转RGB(OpenCV默认BGR, 模型训练时使用RGB)
 * 3. 归一化与张量转换: 像素值归一化到[0,1], 并按CHW格式排列
 *
 * 教学要点:
 * Letterbox是目标检测中常用的缩放方法, 避免图像变形
 * YOLO模型训练时使用RGB输入, 推理时必须保持一致
 * TensorRT要求输入为CHW格式(通道优先), 而非OpenCV的HWC格式
 */

#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "config.hpp"

namespace Preprocessor {

    // Letterbox缩放(保持宽高比的等比缩放加灰边填充)
    // 核心思想: 取宽高方向上较小的缩放比, 等比缩放后在短边两侧填充灰色
    static cv::Mat letterbox(const cv::Mat& img) {
        // 步骤1: 计算缩放比例, 取较小的缩放比确保图像不会被裁剪
        float scaleW = Config::INPUT_WIDTH  / static_cast<float>(img.cols);
        float scaleH = Config::INPUT_HEIGHT / static_cast<float>(img.rows);
        float scale = std::min(scaleW, scaleH);

        // 步骤2: 按比例等比缩放
        int newWidth  = static_cast<int>(img.cols * scale);
        int newHeight = static_cast<int>(img.rows * scale);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

        // 步骤3: 创建目标尺寸画布并用灰色填充
        cv::Mat padded = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
        padded.setTo(Config::LETTERBOX_FILL_COLOR);

        // 步骤4: 计算居中位置并复制缩放后的图像到画布中央
        int dx = (Config::INPUT_WIDTH  - newWidth)  / 2;
        int dy = (Config::INPUT_HEIGHT - newHeight) / 2;
        resized.copyTo(padded(cv::Rect(dx, dy, newWidth, newHeight)));

        return padded;
    }

    // 将OpenCV图像转换为TensorRT输入张量
    // 流程: BGR转RGB, 归一化到[0,1], HWC转CHW
    static std::vector<float> imageToTensor(const cv::Mat& img) {
        // 步骤1: BGR转RGB
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

        // 步骤2: 归一化到[0,1]
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        // 步骤3: 分配输出张量空间(3通道*高度*宽度)
        int tensorSize = 3 * Config::INPUT_HEIGHT * Config::INPUT_WIDTH;
        std::vector<float> tensor(tensorSize);

        // 步骤4: HWC转CHW. 为每个通道创建视图指向输出张量的对应区域
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            channels[c] = cv::Mat(
                Config::INPUT_HEIGHT,
                Config::INPUT_WIDTH,
                CV_32FC1,
                tensor.data() + c * Config::INPUT_HEIGHT * Config::INPUT_WIDTH
            );
        }
        cv::split(rgb, channels);

        return tensor;
    }

} // namespace Preprocessor

#endif // PREPROCESSOR_HPP
