/**
 * @file types.hpp
 * @brief YOLO11 TensorRT 推理数据结构定义
 *
 * 本文件定义了推理过程中使用的核心数据结构: Detection(单个检测结果)
 *
 * 教学要点:
 * 1. 使用struct组织相关数据, 提高代码可读性
 * 2. 将数据结构独立出来, 降低模块间耦合
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <QMetaType>

// 单个检测结果. 坐标系统以原图左上角为原点(0,0), 单位为像素
struct Detection {
    float x;       // 检测框左上角X坐标(像素)
    float y;       // 检测框左上角Y坐标(像素)
    float w;       // 检测框宽度(像素)
    float h;       // 检测框高度(像素)
    float conf;    // 置信度(0.0~1.0), 值越高表示检测结果越可信
    int class_id;  // 类别索引(对应Config::CLASS_NAMES中的索引)

    // 便捷方法: 获取右下角坐标
    float x2() const { return x + w; }
    float y2() const { return y + h; }

    // 便捷方法: 获取检测框面积
    float area() const { return w * h; }
};

// 摄像头信息
struct CameraInfo {
    int id;              // 摄像头ID
    std::string name;    // 摄像头名称
    std::string url;     // RTSP URL 或 文件路径
    std::string type;    // "rtsp", "file", "webcam"
    bool active;         // 是否正在推流

    CameraInfo() : id(0), active(false) {}
    CameraInfo(int i, const std::string& n, const std::string& u, const std::string& t)
        : id(i), name(n), url(u), type(t), active(false) {}
};

// 注册 Detection 为 Qt 元类型，支持跨线程信号槽传递
Q_DECLARE_METATYPE(Detection)
Q_DECLARE_METATYPE(std::vector<Detection>)

// 将检测结果格式化为字符串(调试用)
inline std::string detectionToString(const Detection& det) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "[class=%d, conf=%.3f, box=(%.1f,%.1f,%.1f,%.1f)]",
             det.class_id, det.conf, det.x, det.y, det.w, det.h);
    return std::string(buf);
}

#endif // TYPES_HPP
