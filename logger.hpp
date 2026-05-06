/**
 * @file logger.hpp
 * @brief TensorRT 日志记录器模块
 *
 * TensorRT推理引擎在运行过程中会输出各种级别的信息(DEBUG, INFO, WARNING, ERROR).
 * 本模块实现了一个自定义日志记录器, 用于接管和过滤TensorRT的日志输出.
 *
 * 教学要点:
 * 1. TensorRT要求用户实现nvinfer1::ILogger接口
 * 2. log()函数会被TensorRT内部调用, 输出推理过程中的状态信息
 * 3. 通过severity级别可以过滤输出, 生产环境通常只显示WARNING以上
 */

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include "NvInfer.h"

// TensorRT自定义日志记录器. 继承nvinfer1::ILogger接口
class TrtLogger : public nvinfer1::ILogger {
public:
    // 日志输出回调函数. noexcept是TensorRT的强制要求
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // 只显示WARNING及以上级别的日志
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

#endif // LOGGER_HPP
