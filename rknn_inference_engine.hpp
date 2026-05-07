/**
 * @file rknn_inference_engine.hpp
 * @brief RKNN (Rockchip NPU) 推理引擎实现
 * @note 需要 RKNN SDK 8.30+
 */
#ifndef RKNN_INFERENCE_ENGINE_HPP
#define RKNN_INFERENCE_ENGINE_HPP

#include "inference_engine.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"

// RKNN SDK 头文件 (条件编译)
// #include <rknn_api.h>

// 前向声明
struct rknn_context;

class RknnInferenceEngine : public IInferenceEngine {
public:
    RknnInferenceEngine();
    ~RknnInferenceEngine() override;

    bool init(const std::string& modelPath, void* stream = nullptr) override;
    std::vector<Detection> infer(const uint8_t* imageData, int width, int height,
                                 float confThresh, float nmsThresh) override;
    std::vector<std::vector<Detection>> batchInfer(const std::vector<const uint8_t*>& images,
                                               int width, int height,
                                               float confThresh, float nmsThresh) override;
    EngineType getType() const override { return EngineType::RKNN; }
    std::string getModelName() const override { return modelName_; }
    bool isLoaded() const override { return ctx_ != 0; }
    int getInputSize() const override { return inputSize_; }

private:
    // RKNN 上下文
    rknn_context ctx_ = 0;
    
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<Postprocessor> postprocessor_;
    std::string modelName_;
    int inputSize_ = 640;
    
    // 输入输出 tensor 信息
    int inputIndex_ = 0;
    int outputIndex_ = 0;
};

#endif // RKNN_INFERENCE_ENGINE_HPP