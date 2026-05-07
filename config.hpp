/**
 * @file config.hpp
 * @brief YOLO11 TensorRT 推理配置模块
 *
 * 本文件集中管理所有推理相关的配置参数, 包括:
 * 模型输入尺寸, 置信度与NMS阈值, 检测类别定义, 模型文件与输出路径
 *
 * 教学要点:
 * 1. 使用namespace组织配置项, 避免全局变量污染
 * 2. 使用constexpr在编译期确定常量值(性能更优)
 * 3. 类别数必须与模型训练时的输出一致
 */

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace Config {
    // 模型输入图像宽度(像素). YOLO默认使用640x640
    constexpr int INPUT_WIDTH = 640;

    // 模型输入图像高度(像素)
    constexpr int INPUT_HEIGHT = 640;

    // 置信度阈值(0.0~1.0). 低于此阈值的检测结果将被过滤
    constexpr float CONF_THRESHOLD = 0.25f;

    // NMS(非极大值抑制)IOU阈值(0.0~1.0). 用于去除重叠度高的冗余框
    constexpr float IOU_THRESHOLD = 0.45f;

    // 检测类别总数. 必须与模型训练时的类别数完全一致
    constexpr int NUM_CLASSES = 11;

    // 批量推理配置
    constexpr int BATCH_SIZE = 4;
    constexpr bool USE_BATCH_INFERENCE = true; // 设为true启用批量推理 (提升视频流性能)

    // 类别名称列表. 顺序必须与训练时的类别索引一致
    // 索引0=helmet, 1=gloves, 2=vest, 3=boots, 4=goggles
    // 索引5=none, 6=Person, 7=no_helmet, 8=no_goggle, 9=no_gloves, 10=no_boots
    const std::vector<std::string> CLASS_NAMES = {
        "helmet", "gloves", "vest", "boots", "goggles",
        "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"
    };

    // TensorRT Engine模型路径(相对于可执行文件的路径, 便于分发)
    const std::string MODEL_PATH = "model/your_model.engine";

    // 检测结果输出目录(相对于可执行文件)
    const std::string OUTPUT_DIR = "output";

    // Letterbox填充颜色(BGR格式). 默认灰色(114,114,114), 与YOLO训练时一致
    const cv::Scalar LETTERBOX_FILL_COLOR = cv::Scalar(114, 114, 114);

    // ============ WebSocket 告警推送配置 ============

    // WebSocket 服务端口
    constexpr int WEBSOCKET_PORT = 9090;

    // 本机 IP(仅作为自动获取失败时的回退值, 运行时优先自动获取)
    const std::string HOST_IP = "192.168.124.28";

    // HTTP 文件服务端口(用于 URL 构造)
    constexpr int HTTP_PORT = 9091;

    // MJPEG 实时流服务端口
    constexpr int STREAM_PORT = 9092;

    // ACK 等待超时(毫秒), 超时后重发
    constexpr int ACK_TIMEOUT_MS = 5000;

    // 环形缓冲区帧数(告警前缓存). ~3s @ 30fps
    constexpr int RING_BUFFER_FRAMES = 90;

    // 告警触发后再录制帧数(前后共 ~5s 视频)
    constexpr int ALERT_AFTER_FRAMES = 60;   // ~2s @ 30fps

    // 同一类别告警最小间隔(毫秒)
    constexpr int ALERT_COOLDOWN_MS = 10000;

    // ====== 视频录制配置 ======
    // 默认录像保存根目录
    constexpr const char* RECORD_DIR = "recordings";

    // 录像文件命名格式: {日期}/{摄像头ID}/{起始时间}-{结束时间}.mp4
    // 例如: recordings/20250707/camera_0/102535-104520.mp4

    // 录像删除策略: 0=不删除, 1=按天数, 2=按文件大小(G)
    constexpr int RECORD_DELETE_POLICY = 1;
    constexpr int RECORD_KEEP_DAYS = 7;        // 保留天数
    constexpr float RECORD_MAX_SIZE_GB = 10.0;  // 最大磁盘占用(G)
}

#endif // CONFIG_HPP
