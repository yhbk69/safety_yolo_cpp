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
#include <atomic>
#include <deque>
#include <unordered_map>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "yolo_trt_engine.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "types.hpp"
#include "config.hpp"

namespace Ui { class MainWindow; }


// ============================================================
// InferenceWorker: 后台推理线程
// ============================================================
class InferenceWorker : public QObject {
    Q_OBJECT

public:
    explicit InferenceWorker(YoloTrtEngine* engine);
    ~InferenceWorker() override = default;

public slots:
    void processVideo(const QString& path, float confThresh, float nmsThresh);
    void processCamera(float confThresh, float nmsThresh);
    void stop();

signals:
    void frameProcessed(QImage image, std::vector<Detection> detections, double elapsedMs);
    // 告警视频文件已保存: 视频路径, 截图路径, 告警JSON
    void alertSaved(QString videoPath, QString imagePath, QString alertJson);
    void finished();
    void errorOccurred(const QString& message);

private:
    struct FrameResult {
        QImage image;
        std::vector<Detection> detections;
    };

    FrameResult processOneFrame(const cv::Mat& frame, float confThresh, float nmsThresh);
    void checkAlert(const std::vector<Detection>& detections, const cv::Mat& annotatedFrame);
    void saveAlertFiles(const QString& alarmId, const QString& alarmType);

    YoloTrtEngine* engine_;
    std::atomic<bool> running_{false};

    // 环形缓冲区
    std::mutex bufferMutex_;
    std::deque<cv::Mat> frameBuffer_;

    // 告警冷却
    std::unordered_map<int, std::chrono::steady_clock::time_point> lastAlertTime_;

    // 告警录制状态
    std::atomic<bool> alertRecording_{false};
    int alertRemainingFrames_ = 0;
    std::deque<cv::Mat> alertBuffer_;
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

private slots:
    void onOpenImage();
    void onOpenVideo();
    void onOpenCamera();
    void onOpenFolder();
    void onBrowseModel();
    void onLoadModel();
    void onStopProcessing();
    void onConfThresholdChanged(int value);
    void onNmsThresholdChanged(int value);
    void onBatchInferenceToggled(bool checked);  // 批量推理开关
    void onFrameProcessed(QImage image, std::vector<Detection> detections, double elapsedMs);
    void onWorkerFinished();
    void onWorkerError(const QString& message);

    // 告警
    void onAlertSaved(const QString& videoPath, const QString& imagePath, const QString& alertJson);
    void onWsClientConnected();
    void onWsTextMessage(const QString& message);
    void retryAlarm(const QString& alarmId);

private:
    void setupConnections();
    void updateThresholdLabels();
    void processSingleImage(const std::string& path);
    void updateDisplay(const QImage& image);
    void updateDetectionList(const std::vector<Detection>& detections, double elapsedMs);
    void enableControls(bool enabled);
    void safeStopWorker();
    void startWebSocketServer();
    void startHttpFileServer();
    void closeEvent(QCloseEvent* event) override;

    Ui::MainWindow* ui;
    QLabel* statusMessageLabel_;
    QLabel* fpsLabel_;
    QLabel* timeLabel_;
    QLabel* wsAddressLabel_;

    // 日志输出函数
    void log(const QString& category, const QString& message);
    QString currentTimestamp();

    std::unique_ptr<YoloTrtEngine> engine_;
    QThread*  workerThread_ = nullptr;
    InferenceWorker* worker_ = nullptr;

    QWebSocketServer* wsServer_ = nullptr;
    QList<QWebSocket*> wsClients_;
    QTcpServer* httpServer_ = nullptr;

    // 待确认的告警: alarm_id → {json消息, 重试定时器}
    struct PendingAlarm {
        QString jsonMessage;
        QTimer* retryTimer = nullptr;
    };
    QMap<QString, PendingAlarm> pendingAlarms_;

    bool isProcessing_ = false;
    float confThreshold_;
    float nmsThreshold_;
};

#endif // MAINWINDOW_HPP
