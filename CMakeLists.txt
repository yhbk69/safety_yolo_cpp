# ============================================================
# CMakeLists.txt - YOLO11 TensorRT 推理系统 (Qt GUI 版本)
# ============================================================
# 使用方法:
#   1. 安装 Qt 5.15.2 (选择 MSVC 2019 64-bit 组件)
#   2. 安装 CMake >= 3.16
#   3. 在项目目录下执行:
#      mkdir build && cd build
#      cmake .. -G "Visual Studio 18 2026" -A x64 -DQt5_DIR="C:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5"
#      cmake --build . --config Release
#
# 注意: CMake 会自动处理 Qt 的 MOC (Meta-Object Compiler),
# 不需要额外配置, 比 .vcxproj + Qt VS Tools 更简洁.
# ============================================================

cmake_minimum_required(VERSION 3.16)
project(yolo11_1 LANGUAGES CXX)

# ============ 编译选项 ============
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 UTF-8 源文件编码(解决 Windows 下中文乱码问题)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 自动生成 MOC 文件(处理 Q_OBJECT 宏, Qt 必需)
set(CMAKE_AUTOMOC ON)

# 自动处理 .ui 文件(用 uic 生成 ui_*.h)
set(CMAKE_AUTOUIC ON)

# ============ 查找依赖库 ============

# ---- Qt5 (GUI 框架) ----
# 安装后设置 Qt5_DIR 环境变量, 或在 cmake 时指定:
#   cmake .. -DQt5_DIR="C:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5"
find_package(Qt5 REQUIRED COMPONENTS Widgets WebSockets)

# ---- OpenCV (图像处理) ----
set(OpenCV_DIR "C:/opencv412/opencv/build")
find_package(OpenCV REQUIRED)

# ---- CUDA (GPU 加速) ----
find_package(CUDA REQUIRED)

# ---- TensorRT (推理引擎) ----
# TensorRT 没有官方的 CMake 配置文件, 手动指定路径
set(TENSORRT_DIR "C:/tensorRT10/TensorRT-10.1.0.27")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")

# ============ 源文件列表 ============
set(SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.ui        # Qt UI 布局文件(用 Qt Creator 可视化编辑)
)

set(HEADERS
    mainwindow.hpp         # 含 Q_OBJECT 宏, 需要 MOC 处理
    config.hpp
    types.hpp
    logger.hpp
    preprocessor.hpp
    postprocessor.hpp
    yolo_trt_engine.hpp
    inference_runner.hpp
)

# ============ 生成可执行文件 ============
add_executable(yolo11_1 WIN32 ${SOURCES} ${HEADERS})

# ============ 头文件路径 ============
target_include_directories(yolo11_1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}          # 项目根目录(自身头文件)
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# ============ 链接库 ============
# 指定 TensorRT 库搜索路径(让链接器能找到 nvinfer_10.lib)
target_link_directories(yolo11_1 PRIVATE ${TENSORRT_LIB_DIR})

target_link_libraries(yolo11_1 PRIVATE
    Qt5::Widgets
    Qt5::WebSockets
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer_10          # TensorRT 推理库
    nvonnxparser_10     # TensorRT ONNX 解析器
)

# ============ 输出信息 ============
message(STATUS "Qt5 found: ${Qt5Widgets_INCLUDE_DIRS}")
message(STATUS "OpenCV found: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA found: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
