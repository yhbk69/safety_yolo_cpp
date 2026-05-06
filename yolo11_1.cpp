/**
 * @file yolo11_1.cpp
 * @brief YOLO11 TensorRT 推理系统 - 主入口文件
 *
 * 本程序是一个基于TensorRT的YOLO11目标检测推理系统,
 * 用于实时检测个人防护装备(PPE)的佩戴情况.
 *
 * 代码采用模块化设计, 各文件职责如下:
 * config.hpp           - 配置参数(阈值, 类别, 路径等)
 * logger.hpp           - TensorRT日志记录器
 * types.hpp            - 数据结构定义(Detection)
 * preprocessor.hpp     - 图像预处理(Letterbox, 归一化, CHW转换)
 * postprocessor.hpp    - 后处理(解码, NMS, 绘制结果)
 * yolo_trt_engine.hpp  - TensorRT推理引擎(GPU内存管理, 推理)
 * inference_runner.hpp - 推理执行器(图片/视频/摄像头/文件夹)
 * yolo11_1.cpp         - 主入口(命令行解析, 调度执行)
 *
 * 使用方式:
 * yolo11_1.exe image.jpg              推理单张图片
 * yolo11_1.exe video.mp4              推理视频文件
 * yolo11_1.exe camera                 实时摄像头推理
 * yolo11_1.exe images <folder_path>   批量处理文件夹
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

#include "config.hpp"
#include "types.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "yolo_trt_engine.hpp"
#include "inference_runner.hpp"

namespace fs = std::filesystem;

// 打印程序使用帮助信息
void printUsage(const char* programName) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "          使用帮助" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n支持的输入模式:" << std::endl;
    std::cout << "  1. 单张图片推理" << std::endl;
    std::cout << "     " << programName << " image.jpg" << std::endl;
    std::cout << "\n  2. 视频文件推理" << std::endl;
    std::cout << "     " << programName << " video.mp4" << std::endl;
    std::cout << "\n  3. 实时摄像头推理" << std::endl;
    std::cout << "     " << programName << " camera" << std::endl;
    std::cout << "\n  4. 批量处理文件夹" << std::endl;
    std::cout << "     " << programName << " images <folder_path>" << std::endl;
    std::cout << "\n支持的图片格式: .jpg, .jpeg, .png, .bmp" << std::endl;
    std::cout << "支持的视频格式: .mp4, .avi, .mov, .mkv" << std::endl;
    std::cout << "\n按ESC键可退出推理窗口" << std::endl;
}

// 判断文件扩展名是否匹配目标集合
bool hasExtension(const std::string& filePath, const std::vector<std::string>& extensions) {
    std::string ext = fs::path(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

// 程序主入口
int main(int argc, char** argv) {
    // 将控制台编码设为 UTF-8（解决 Windows 下中文乱码）
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // 启动信息
    std::cout << "========================================" << std::endl;
    std::cout << "   YOLO11 TensorRT C++ 推理系统" << std::endl;
    std::cout << "   PPE 个人防护装备检测" << std::endl;
    std::cout << "========================================" << std::endl;

    // 创建输出目录
    fs::create_directories(Config::OUTPUT_DIR);
    std::cout << "[Main] Output directory: " << Config::OUTPUT_DIR << std::endl;

    // 加载模型
    std::cout << "[Main] Loading model: " << Config::MODEL_PATH << std::endl;
    InferenceRunner runner(Config::MODEL_PATH);
    std::cout << "[Main] Model loaded successfully!" << std::endl;

    // 命令行参数解析
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string mode = argv[1];

    // 模式调度
    try {
        if (mode == "camera") {
            std::cout << "\n[Main] Mode: Camera" << std::endl;
            runner.runCamera();

        } else if (mode == "images" && argc >= 3) {
            std::cout << "\n[Main] Mode: Folder batch - " << argv[2] << std::endl;
            runner.runFolder(argv[2]);

        } else {
            std::string path = argv[1];
            std::cout << "\n[Main] Mode: Single file - " << path << std::endl;

            std::vector<std::string> imageExts = {".jpg", ".jpeg", ".png", ".bmp"};
            std::vector<std::string> videoExts = {".mp4", ".avi", ".mov", ".mkv"};

            if (hasExtension(path, imageExts)) {
                runner.runImage(path);
            } else if (hasExtension(path, videoExts)) {
                runner.runVideo(path);
            } else {
                std::cerr << "[Main] Error: Unsupported format: "
                          << fs::path(path).extension().string() << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "\n[Main] Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n[Main] Done!" << std::endl;
    return 0;
}
