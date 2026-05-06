/**
 * @file yolo_trt_engine.hpp
 * @brief YOLO11 TensorRT 推理引擎模块
 *
 * 本模块是推理系统的核心, 封装了TensorRT引擎的加载, GPU内存管理和推理执行.
 *
 * TensorRT推理流程概览:
 * 1. 加载.engine文件, 反序列化得到ICudaEngine
 * 2. 创建执行上下文IExecutionContext
 * 3. 分配GPU输入/输出缓冲区
 * 4. 推理循环: CPU到GPU复制, executeV2推理, GPU到CPU复制, 后处理
 * 5. 释放GPU内存和TensorRT对象
 *
 * 教学要点:
 * RAII模式: 构造函数初始化, 析构函数清理, 确保资源不会泄漏
 * TensorRT 10.x使用智能指针管理对象生命周期, 不再需要手动destroy()
 * CUDA内存操作是性能关键, 应尽量减少CPU-GPU数据传输
 */

#ifndef YOLO_TRT_ENGINE_HPP
#define YOLO_TRT_ENGINE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "logger.hpp"
#include "config.hpp"
#include "types.hpp"
#include "postprocessor.hpp"

// YOLO TensorRT推理引擎类
// 使用RAII模式管理资源
class YoloTrtEngine {
public:
    // 构造函数: 加载TensorRT Engine并初始化GPU资源
    explicit YoloTrtEngine(const std::string& enginePath) {
        loadEngine(enginePath);
        createContext();
        allocateGpuBuffers();
        std::cout << "[Engine] TensorRT engine loaded successfully" << std::endl;
    }

    // 析构函数: 释放所有GPU资源. TensorRT 10.x使用智能指针, 无需手动destroy()
    ~YoloTrtEngine() {
        cudaFree(gpuInputBuffer_);
        cudaFree(gpuOutputBuffer_);
    }

    // 禁止拷贝(GPU资源不能被共享)
    YoloTrtEngine(const YoloTrtEngine&) = delete;
    YoloTrtEngine& operator=(const YoloTrtEngine&) = delete;

    // 允许移动语义(资源转移)
    YoloTrtEngine(YoloTrtEngine&&) = default;
    YoloTrtEngine& operator=(YoloTrtEngine&&) = default;

    // 执行一次推理
    // confThreshold: 置信度阈值(覆盖Config默认值)
    // iouThreshold:  NMS的IOU阈值(覆盖Config默认值)
    void infer(const std::vector<float>& input, std::vector<Detection>& detections,
               int imgWidth, int imgHeight,
               float confThreshold = Config::CONF_THRESHOLD,
               float iouThreshold = Config::IOU_THRESHOLD) {
        // 步骤1: 将输入数据从CPU复制到GPU
        cudaMemcpy(gpuInputBuffer_, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        // 步骤2: 执行TensorRT推理
        context_->executeV2(gpuBuffers_);

        // 步骤3: 将输出数据从GPU复制回CPU
        std::vector<float> output(getOutputSize());
        cudaMemcpy(output.data(), gpuOutputBuffer_, output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // 步骤4: 后处理(解码加NMS), 传入动态阈值
        detections = Postprocessor::decodeDetections(output.data(), imgWidth, imgHeight,
                                                      confThreshold, iouThreshold);
    }

    // 获取输入张量的大小(元素个数) = 宽度*高度*3
    int getInputSize() const {
        return Config::INPUT_WIDTH * Config::INPUT_HEIGHT * 3;
    }

    // 获取输出张量的大小(元素个数) = (类别数+4)*8400
    int getOutputSize() const {
        return (Config::NUM_CLASSES + 4) * 8400;
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;          // TensorRT运行时对象
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;         // TensorRT推理引擎
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  // 执行上下文
    void* gpuInputBuffer_ = nullptr;    // GPU输入缓冲区
    void* gpuOutputBuffer_ = nullptr;   // GPU输出缓冲区
    void* gpuBuffers_[2] = {nullptr, nullptr};  // GPU缓冲区指针数组
    TrtLogger logger_;                  // TensorRT日志记录器

    // 从磁盘加载并反序列化TensorRT Engine
    void loadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("[Engine] Cannot open engine file: " + path);
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> trtModelData(size);
        file.read(trtModelData.data(), size);
        file.close();

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger_));
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(trtModelData.data(), size));

        if (!engine_) {
            throw std::runtime_error("[Engine] Failed to deserialize engine from: " + path);
        }
    }

    // 创建推理执行上下文
    void createContext() {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
    }

    // 分配GPU内存缓冲区
    void allocateGpuBuffers() {
        cudaMalloc(&gpuInputBuffer_,  getInputSize()  * sizeof(float));
        cudaMalloc(&gpuOutputBuffer_, getOutputSize() * sizeof(float));
        gpuBuffers_[0] = gpuInputBuffer_;
        gpuBuffers_[1] = gpuOutputBuffer_;
    }
};

#endif // YOLO_TRT_ENGINE_HPP
