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
#include <stdexcept>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "logger.hpp"
#include "config.hpp"
#include "types.hpp"
#include "postprocessor.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("[CUDA Error] ") + \
                cudaGetErrorName(err) + ": " + cudaGetErrorString(err)); \
        } \
    } while(0)

class YoloTrtEngine {
public:
    explicit YoloTrtEngine(const std::string& enginePath) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        loadEngine(enginePath);
        createContext();
        allocateGpuBuffers();
        std::cout << "[Engine] TensorRT engine loaded successfully" << std::endl;
    }

    ~YoloTrtEngine() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
        cudaFree(gpuInputBuffer_);
        cudaFree(gpuOutputBuffer_);
    }

    YoloTrtEngine(const YoloTrtEngine&) = delete;
    YoloTrtEngine& operator=(const YoloTrtEngine&) = delete;
    YoloTrtEngine(YoloTrtEngine&&) = default;
    YoloTrtEngine& operator=(YoloTrtEngine&&) = default;

    void infer(const std::vector<float>& input, std::vector<Detection>& detections,
               int imgWidth, int imgHeight,
               float confThreshold = Config::CONF_THRESHOLD,
               float iouThreshold = Config::IOU_THRESHOLD) {
        CUDA_CHECK(cudaMemcpyAsync(gpuInputBuffer_, input.data(),
                    input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        context_->executeV2(gpuBuffers_);

        std::vector<float> output(getOutputSize());
        CUDA_CHECK(cudaMemcpy(output.data(), gpuOutputBuffer_,
                    output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        detections = Postprocessor::decodeDetections(output.data(), imgWidth, imgHeight,
                                                       confThreshold, iouThreshold);
    }

    int getInputSize() const {
        return Config::INPUT_WIDTH * Config::INPUT_HEIGHT * 3;
    }

    int getOutputSize() const {
        return (Config::NUM_CLASSES + 4) * 8400;
    }

    // 批量推理: 输入batch*N, 输出每个frame的detections
    // 注意: engine必须以对应BATCH_SIZE构建, 否则推理结果不正确
    void batchInfer(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<Detection>>& detectionsList,
                  const std::vector<std::pair<int,int>>& imgSizes,
                  float confThreshold = Config::CONF_THRESHOLD,
                  float iouThreshold = Config::IOU_THRESHOLD) {
        const int batchSize = static_cast<int>(inputs.size());
        if (batchSize == 0) return;
        if (batchSize > Config::BATCH_SIZE) {
            throw std::runtime_error("[Engine] Batch size " + std::to_string(batchSize) +
                " exceeds configured maximum " + std::to_string(Config::BATCH_SIZE));
        }

        // 一次性拷贝整个batch到GPU
        size_t batchInputSize = inputs[0].size() * batchSize;
        std::vector<float> batchInput(batchInputSize);
        for (int i = 0; i < batchSize; ++i) {
            std::copy(inputs[i].begin(), inputs[i].end(), batchInput.begin() + i * inputs[i].size());
        }

        CUDA_CHECK(cudaMemcpyAsync(gpuInputBuffer_, batchInput.data(),
                    batchInputSize * sizeof(float), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        context_->executeV2(gpuBuffers_);

        std::vector<float> output(getOutputSize() * batchSize);
        CUDA_CHECK(cudaMemcpy(output.data(), gpuOutputBuffer_,
                    output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // 逐帧后处理
        detectionsList.resize(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            float* frameOutput = output.data() + i * getOutputSize();
            detectionsList[i] = Postprocessor::decodeDetections(frameOutput, imgSizes[i].first, imgSizes[i].second,
                                                     confThreshold, iouThreshold);
        }
    }

private:
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    void* gpuInputBuffer_ = nullptr;
    void* gpuOutputBuffer_ = nullptr;
    void* gpuBuffers_[2] = {nullptr, nullptr};
    TrtLogger logger_;

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

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(trtModelData.data(), size));
        if (!engine_) {
            throw std::runtime_error("[Engine] Failed to deserialize engine from: " + path);
        }
    }

    void createContext() {
        context_.reset(engine_->createExecutionContext());
    }

    void allocateGpuBuffers() {
        // 按最大批次分配GPU内存, 避免batchInfer时越界写显存
        CUDA_CHECK(cudaMalloc(&gpuInputBuffer_,  getInputSize()  * Config::BATCH_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpuOutputBuffer_, getOutputSize() * Config::BATCH_SIZE * sizeof(float)));
        gpuBuffers_[0] = gpuInputBuffer_;
        gpuBuffers_[1] = gpuOutputBuffer_;
    }
};

#undef CUDA_CHECK

#endif // YOLO_TRT_ENGINE_HPP
