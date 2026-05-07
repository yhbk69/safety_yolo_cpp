#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QMainWindow>
#include <QStatusBar>
#include <QThread>
#include <QImage>
#include <QLabel>
#include <QWebSocketServer>
#include <QWebSocket>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>
#include <QDir>
#include <QTcpServer>
#include <QTcpSocket>
#include <QDialog>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QDesktopServices>
#include <QSharedPointer>
#include <atomic>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <memory>

#include <opencv2/opencv.hpp>

#include "yolo_trt_engine.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "types.hpp"
#include "config.hpp"
#include "runtime_config.hpp"

namespace Ui { class MainWindow; }


// ============================================================
// InferenceWorker: 后台推理线程
// ============================================================
class InferenceWorker : public QObject {
    Q_OBJECT

public:
    explicit InferenceWorker(YoloTrtEngine* engine, int cameraId = 0,
                             const QString& cameraName = "camera_0",
                             const QString& source = "");
    ~InferenceWorker() override = default;

    int cameraId() const { return cameraId_; }
    QString cameraName() const { return cameraName_; }

public slots:
    void processVideo(const QString& path, float confThresh, float nmsThresh);
    void processCamera(float confThresh, float nmsThresh);
    void processSource(float confThresh, float nmsThresh);
    void stop();
    void setBatchInference(bool enabled) { useBatchInference_ = enabled; }

signals:
    void frameProcessed(int cameraId, QImage image, std::vector<Detection> detections, double elapsedMs);
    // 告警视频文件已保存: 视频路径, 截图路径, 告警JSON
    void alertSaved(int cameraId, QString videoPath, QString imagePath, QString alertJson);
    void finished(int cameraId);
    void errorOccurred(int cameraId, const QString& message);

private:
    struct FrameResult {
        QImage image;
        std::vector<Detection> detections;
    };

    FrameResult processOneFrame(const cv::Mat& frame, float confThresh, float nmsThresh);
    void checkAlert(const std::vector<Detection>& detections, const std::shared_ptr<cv::Mat>& annotatedFrame);
    void saveAlertFiles(const QString& alarmId, const QString& alarmType);

    YoloTrtEngine* engine_;
    int cameraId_;
    QString cameraName_;
    QString source_;  // 摄像头源: 数字=设备ID, rtsp://=RTSP流, 空=使用cameraId_
    std::atomic<bool> running_{false};

    // 批量推理状态
    std::atomic<bool> useBatchInference_{false};
    std::vector<std::vector<float>> batchTensors_;
    std::vector<std::pair<int,int>> batchImgSizes_;
    int batchCounter_ = 0;

    // 环形缓冲区(共享指针避免深拷贝)
    std::mutex bufferMutex_;
    std::deque<std::shared_ptr<cv::Mat>> frameBuffer_;

    // 告警冷却
    std::unordered_map<int, std::chrono::steady_clock::time_point> lastAlertTime_;

    // 告警录制状态
    std::atomic<bool> alertRecording_{false};
    int alertRemainingFrames_ = 0;
    std::deque<std::shared_ptr<cv::Mat>> alertBuffer_;
    QString pendingAlarmType_;

    // 告警输出目录
    QString outputDir_;
};


// ============================================================
// MainWindow: 主窗口
// ============================================================
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    static QString getHostIp();

private slots:
    void onOpenImage();
    void onOpenVideo();
    void onOpenCamera(bool checked);
    void onAddCamera();
    void onRemoveCamera(int cameraId);
    void onOpenFolder();
    void onBrowseModel();
    void onLoadModel();
    void onReloadModel();
    void onStopProcessing();
    void onConfThresholdChanged(int value);
    void onNmsThresholdChanged(int value);
    void onBatchInferenceToggled(bool checked);
    void onSettings();
    void onFrameProcessed(int cameraId, QImage image, std::vector<Detection> detections, double elapsedMs);
    void onWorkerFinished(int cameraId);
    void onWorkerError(int cameraId, const QString& message);

    // 告警
    void onAlertSaved(int cameraId, const QString& videoPath, const QString& imagePath, const QString& alertJson);
    void onWsClientConnected();
    void onWsTextMessage(const QString& message);
    void retryAlarm(const QString& alarmId);

    // WebSocket 消息处理
    void handleSyncRequest(const QString& lastAlarmId);
    void handleGetStreams();
    void handleSetFence(const QString& streamId, const QJsonObject& fence);
    void handleViewStream(const QString& streamId);

private:
    void setupConnections();
    void updateThresholdLabels();
    void processSingleImage(const std::string& path);
    void updateDisplay(const QImage& image);
    void updateDetectionList(const std::vector<Detection>& detections, double elapsedMs);
    void enableControls(bool enabled);
    void stopCamera(int cameraId);
    void stopAllCameras();
    void startWebSocketServer();
    void startHttpFileServer();
    void startMjpegServer();
    void pushMjpegFrame(const QByteArray& jpegData);
    void closeEvent(QCloseEvent* event) override;
    void loadRuntimeConfig();
    void saveRuntimeConfig();

    // 多摄像头管理
    struct CameraWorker {
        QThread* thread = nullptr;
        InferenceWorker* worker = nullptr;
        QLabel* displayLabel = nullptr;  // 该路对应的显示标签
    };
    QMap<int, CameraWorker> cameraWorkers_;
    int nextCameraId_ = 1;  // 下一个可用的摄像头ID(0保留给默认摄像头按钮)
    int activeDisplayCamera_ = 0;  // 当前显示画面的摄像头ID

    Ui::MainWindow* ui;
    QLabel* statusMessageLabel_;
    QLabel* fpsLabel_;
    QLabel* timeLabel_;
    QLabel* wsAddressLabel_;

    // 日志输出函数
    void log(const QString& category, const QString& message);
    QString currentTimestamp();

    std::unique_ptr<YoloTrtEngine> engine_;

    QWebSocketServer* wsServer_ = nullptr;
    QList<QWebSocket*> wsClients_;
    QTcpServer* httpServer_ = nullptr;
    QTcpServer* mjpegServer_ = nullptr;
    QList<QTcpSocket*> mjpegClients_;

    // MJPEG最新帧(线程安全)
    std::mutex mjpegMutex_;
    QByteArray mjpegFrame_;
    std::atomic<bool> mjpegKeyFrame_{false};

    // MJPEG帧率控制
    QTimer* mjpegFrameTimer_ = nullptr;
    QByteArray mjpegPendingFrame_;

    // 待确认的告警: alarm_id → {json消息, 重试定时器, 重试次数}
    struct PendingAlarm {
        QString jsonMessage;
        QTimer* retryTimer = nullptr;
        int retryCount = 0;
    };
    QMap<QString, PendingAlarm> pendingAlarms_;

    static constexpr int MAX_RETRY_COUNT = 10;

    // 围栏设置: stream_id → fence区域
    struct FenceRegion {
        float x1, y1, x2, y2;
    };
    QMap<QString, FenceRegion> fenceRegions_;

    // ====== 视频录制 ======
    struct CameraRecording {
        int cameraId = 0;
        QString videoPath;        // 当前录像文件路径
        cv::VideoWriter* writer = nullptr;  // OpenCV录像 writer
        bool isRecording = false;  // 改为普通bool
        QDateTime startTime;   // 开始时间
    };
    QMap<int, CameraRecording*> cameraRecordings_;  // 用原始指针
    QString getRecordDir(int cameraId);  // 获取录像目录
    void writeRecordingFrame(int cameraId, const cv::Mat& frame);  // 写帧到录像

private slots:
    // 录制相关
    void onStartRecording();
    void onStopRecording();
    void onViewRecordings();
    void onClearOldRecordings();

private:
    bool isProcessing_ = false;  // 视频模式用
    float confThreshold_;
    float nmsThreshold_;
};

#endif // MAINWINDOW_HPP
