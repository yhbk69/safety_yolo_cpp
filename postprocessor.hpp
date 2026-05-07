/**
 * @file postprocessor.hpp
 * @brief YOLO11 后处理与可视化模块
 *
 * 本模块负责处理TensorRT模型输出的原始张量, 包括:
 * 1. 解析模型输出, 提取检测框和类别信息
 * 2. NMS(非极大值抑制)去除冗余框
 * 3. 在图像上绘制检测结果
 *
 * 教学要点:
 * YOLO11的输出格式为[num_classes+4, 8400](列优先)
 * NMS是目标检测中必不可少的步骤, 用于去除重叠的冗余框
 */

#ifndef POSTPROCESSOR_HPP
#define POSTPROCESSOR_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "config.hpp"
#include "types.hpp"

namespace Postprocessor {

    // 从模型原始输出中提取有效检测结果
    // YOLO11 TensorRT Engine输出格式: [channels, 8400]列优先(column-major)
    // 通道0=所有cx, 通道1=所有cy, 通道2=所有w, 通道3=所有h, 通道4起=类别分数
    // confThreshold: 置信度阈值, 低于该值的检测被过滤
    // iouThreshold:  NMS的IOU阈值, 用于去除重叠框
    static std::vector<Detection> decodeDetections(const float* output, int imgWidth, int imgHeight,
                                                    float confThreshold = Config::CONF_THRESHOLD,
                                                    float iouThreshold = Config::IOU_THRESHOLD) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        // 计算缩放比例(用于将模型输出坐标映射回原图)
        float scaleX = static_cast<float>(imgWidth)  / Config::INPUT_WIDTH;
        float scaleY = static_cast<float>(imgHeight) / Config::INPUT_HEIGHT;

        constexpr int numAnchors = 8400;
        constexpr int numClasses = Config::NUM_CLASSES;

        // 遍历所有8400个锚点, 找出置信度高于阈值的目标
        for (int i = 0; i < numAnchors; ++i) {
            // 找出最高分类得分及其类别索引
            int classId = 0;
            float maxConf = -1.0f;
            for (int j = 0; j < numClasses; ++j) {
                float score = output[(4 + j) * numAnchors + i];
                if (score > maxConf) {
                    maxConf = score;
                    classId = j;
                }
            }

            // 过滤低置信度检测
            if (maxConf < confThreshold) continue;

            // 读取边界框坐标(按列优先)
            float cx = output[0 * numAnchors + i];
            float cy = output[1 * numAnchors + i];
            float w  = output[2 * numAnchors + i];
            float h  = output[3 * numAnchors + i];

            // 将坐标缩放回原始图像尺寸
            float x = (cx - w / 2.0f) * scaleX;
            float y = (cy - h / 2.0f) * scaleY;
            float boxW = w * scaleX;
            float boxH = h * scaleY;

            // 确保边界框不超出图像范围
            int boxX = std::max(0, static_cast<int>(x));
            int boxY = std::max(0, static_cast<int>(y));
            int boxW_int = std::min(imgWidth - boxX, static_cast<int>(boxW));
            int boxH_int = std::min(imgHeight - boxY, static_cast<int>(boxH));

            boxes.emplace_back(boxX, boxY, boxW_int, boxH_int);
            confidences.push_back(maxConf);
            classIds.push_back(classId);
        }

        // NMS(非极大值抑制)去除冗余框
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

        // 将保留的框转换为Detection结构体
        std::vector<Detection> detections;
        detections.reserve(indices.size());
        for (int idx : indices) {
            Detection det;
            det.x = static_cast<float>(boxes[idx].x);
            det.y = static_cast<float>(boxes[idx].y);
            det.w = static_cast<float>(boxes[idx].width);
            det.h = static_cast<float>(boxes[idx].height);
            det.conf = confidences[idx];
            det.class_id = classIds[idx];
            detections.push_back(det);
        }

        return detections;
    }

    // 在图像上绘制检测结果(边界框和类别标签)
    static void drawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            std::string className = Config::CLASS_NAMES[det.class_id];
            // no_xx 用红色，正常用绿色
            cv::Scalar boxColor = (className.find("no_") == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            char confBuf[8];
            snprintf(confBuf, sizeof(confBuf), "%.3f", det.conf);
            std::string label = className + ": " + confBuf;

            // 绘制边界框
            cv::rectangle(img,
                cv::Point(static_cast<int>(det.x), static_cast<int>(det.y)),
                cv::Point(static_cast<int>(det.x + det.w), static_cast<int>(det.y + det.h)),
                boxColor, 2);

            // 计算文本尺寸
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // 绘制文本背景
            cv::rectangle(img,
                cv::Point(static_cast<int>(det.x), static_cast<int>(det.y) - textSize.height - 5),
                cv::Point(static_cast<int>(det.x) + textSize.width, static_cast<int>(det.y)),
                boxColor, -1);

            // 绘制文本
            cv::putText(img, label,
                cv::Point(static_cast<int>(det.x), static_cast<int>(det.y) - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

} // namespace Postprocessor

#endif // POSTPROCESSOR_HPP
