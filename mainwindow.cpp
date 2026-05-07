/**
 * @file mainwindow.cpp
 * @brief YOLO11 TensorRT 推理系统 - 主窗口实现
 */

#include "mainwindow.hpp"
#include "ui_mainwindow.h"

#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QCloseEvent>
#include <QScreen>
#include <QPixmap>
#include <QListWidgetItem>
#include <QDir>
#include <QNetworkInterface>
#include <QUuid>
#include <QFile>
#include <QInputDialog>
#include <QBuffer>

#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;


// ============================================================
// InferenceWorker
// ============================================================

InferenceWorker::InferenceWorker(YoloTrtEngine* engine, int cameraId,
                                     const QString& cameraName,
                                     const QString& source)
    : engine_(engine)
    , cameraId_(cameraId)
    , cameraName_(cameraName)
    , source_(source)
{
    outputDir_ = QDir::cleanPath(QDir::currentPath() + "/" +
        QString::fromStdString(Config::OUTPUT_DIR));
    QDir().mkpath(outputDir_);
}

void InferenceWorker::processVideo(const QString& path, float confThresh, float nmsThresh) {
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) {
        emit errorOccurred(cameraId_, "无法打开视频文件: " + path);
        emit finished(cameraId_);
        return;
    }
    running_ = true;
    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        auto result = processOneFrame(frame, confThresh, nmsThresh);
        emit frameProcessed(cameraId_, result.image, result.detections, 0);
    }
    cap.release();
    emit finished(cameraId_);
}

void InferenceWorker::processCamera(float confThresh, float nmsThresh) {
    // 使用source_或cameraId_打开摄像头
    cv::VideoCapture cap;
    if (!source_.isEmpty() && source_.startsWith("rtsp://")) {
        cap.open(source_.toStdString());
    } else if (!source_.isEmpty()) {
        bool ok = false;
        int devId = source_.toInt(&ok);
        cap.open(ok ? devId : 0, cv::CAP_DSHOW);
    } else {
        cap.open(cameraId_, cv::CAP_DSHOW);
    }

    if (!cap.isOpened()) {
        emit errorOccurred(cameraId_, "无法打开摄像头, 请检查设备连接");
        emit finished(cameraId_);
        return;
    }
    running_ = true;
    const int frameDelayMs = 33;
    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            if (running_) {
                emit errorOccurred(cameraId_, "摄像头读取失败，设备可能已断开");
            }
            break;
        }
        auto result = processOneFrame(frame, confThresh, nmsThresh);
        emit frameProcessed(cameraId_, result.image, result.detections, 0);
        QThread::msleep(frameDelayMs);
    }
    cap.release();
    emit finished(cameraId_);
}

void InferenceWorker::processSource(float confThresh, float nmsThresh) {
    // 通用源处理, 自动判断类型
    if (!source_.isEmpty() && source_.startsWith("rtsp://")) {
        cv::VideoCapture cap(source_.toStdString());
        if (!cap.isOpened()) {
            emit errorOccurred(cameraId_, "无法打开RTSP流: " + source_);
            emit finished(cameraId_);
            return;
        }
        running_ = true;
        while (running_) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            auto result = processOneFrame(frame, confThresh, nmsThresh);
            emit frameProcessed(cameraId_, result.image, result.detections, 0);
            QThread::msleep(10);
        }
        cap.release();
        emit finished(cameraId_);
    } else {
        processCamera(confThresh, nmsThresh);
    }
}

void InferenceWorker::stop() {
    running_ = false;
}

InferenceWorker::FrameResult InferenceWorker::processOneFrame(
    const cv::Mat& frame, float confThresh, float nmsThresh)
{
    cv::Mat processed = Preprocessor::letterbox(frame);
    std::vector<float> tensor = Preprocessor::imageToTensor(processed);

    std::vector<Detection> detections;

    if (useBatchInference_ && Config::BATCH_SIZE > 1) {
        // 批量推理: 收集帧, 满batch时执行
        batchTensors_.push_back(std::move(tensor));
        batchImgSizes_.emplace_back(frame.cols, frame.rows);
        batchCounter_++;

        if (batchCounter_ >= Config::BATCH_SIZE) {
            std::vector<std::vector<Detection>> batchDetections;
            engine_->batchInfer(batchTensors_, batchDetections, batchImgSizes_,
                               confThresh, nmsThresh);
            // 取最后一帧的检测结果(当前帧)
            if (!batchDetections.empty()) {
                detections = std::move(batchDetections.back());
            }
            batchTensors_.clear();
            batchImgSizes_.clear();
            batchCounter_ = 0;
        } else {
            // 未满batch, 用上一帧的检测结果占位(或跳过)
            // 返回空检测, 画面保持上一帧
            FrameResult result;
            auto displayImg = std::make_shared<cv::Mat>(frame.clone());
            Postprocessor::drawDetections(*displayImg, detections);
            cv::cvtColor(*displayImg, *displayImg, cv::COLOR_BGR2RGB);
            QImage qimg(displayImg->data, displayImg->cols, displayImg->rows,
                        displayImg->step, QImage::Format_RGB888);
            result.image = qimg.copy();
            result.detections = std::move(detections);
            return result;
        }
    } else {
        // 单帧推理
        engine_->infer(tensor, detections, frame.cols, frame.rows, confThresh, nmsThresh);
    }

    // 在原始帧上绘制检测框(后续环形缓冲区/告警共享此Mat)
    auto displayImg = std::make_shared<cv::Mat>(frame.clone());
    Postprocessor::drawDetections(*displayImg, detections);

    // 直接在displayImg上做BGR2RGB, 省掉中间Mat分配
    cv::cvtColor(*displayImg, *displayImg, cv::COLOR_BGR2RGB);
    QImage qimg(displayImg->data, displayImg->cols, displayImg->rows,
                displayImg->step, QImage::Format_RGB888);

    // 环形缓冲区(共享指针, 零拷贝)
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        frameBuffer_.push_back(displayImg);
        if (frameBuffer_.size() > Config::RING_BUFFER_FRAMES) {
            frameBuffer_.pop_front();
        }
    }

    // 告警检测(传共享指针)
    checkAlert(detections, displayImg);

    // 告警录制后续帧(共享指针, 零拷贝)
    if (alertRecording_ && alertRemainingFrames_ > 0) {
        alertBuffer_.push_back(displayImg);
        alertRemainingFrames_--;
        if (alertRemainingFrames_ == 0) {
            saveAlertFiles(QUuid::createUuid().toString(QUuid::WithoutBraces),
                          pendingAlarmType_);
            alertRecording_ = false;
            alertBuffer_.clear();
            pendingAlarmType_.clear();
        }
    }

    FrameResult result;
    result.image = qimg.copy();  // 必须copy: displayImg是共享的, RGB数据会被后续帧覆盖
    result.detections = std::move(detections);
    return result;
}

void InferenceWorker::checkAlert(
    const std::vector<Detection>& detections, const std::shared_ptr<cv::Mat>& annotatedFrame)
{
    for (const auto& det : detections) {
        const auto& name = Config::CLASS_NAMES[det.class_id];
        if (name.find("no_") != 0) continue;

        auto now = std::chrono::steady_clock::now();
        auto it = lastAlertTime_.find(det.class_id);
        if (it != lastAlertTime_.end()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second).count();
            if (elapsed < Config::ALERT_COOLDOWN_MS) continue;
        }
        lastAlertTime_[det.class_id] = now;

        // 启动告警录制
        alertRecording_ = true;
        alertRemainingFrames_ = Config::ALERT_AFTER_FRAMES;
        pendingAlarmType_ = QString::fromStdString(name);

        {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            alertBuffer_ = frameBuffer_;  // 共享指针拷贝, 引用计数+1, 零内存拷贝
        }
        alertBuffer_.push_back(annotatedFrame);

        return;
    }
}

void InferenceWorker::saveAlertFiles(const QString& alarmId, const QString& alarmType) {
    if (alertBuffer_.empty()) return;

    // 告警视频需要BGR格式, 但环形缓冲区已转为RGB, 需要转回BGR保存
    auto toBgr = [](const std::shared_ptr<cv::Mat>& rgbFrame) -> cv::Mat {
        cv::Mat bgr;
        cv::cvtColor(*rgbFrame, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    };

    int frameW = alertBuffer_.back()->cols;
    int frameH = alertBuffer_.back()->rows;
    if (frameW <= 0 || frameH <= 0) return;

    // 文件名: alarm_<alarmId>_<type>
    QString baseName = QString("alarm_%1_%2").arg(alarmId).arg(alarmType);
    QString videoPath = outputDir_ + "/" + baseName + ".mp4";
    QString imagePath = outputDir_ + "/" + baseName + ".jpg";

    // 保存视频(MP4) - 需要BGR格式
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(videoPath.toStdString(), fourcc, 30.0,
                           cv::Size(frameW, frameH));
    if (writer.isOpened()) {
        for (auto& f : alertBuffer_) {
            writer.write(toBgr(f));
        }
        writer.release();
    }

    // 保存截图(第一帧) - 需要BGR格式
    cv::imwrite(imagePath.toStdString(), toBgr(alertBuffer_.front()));

    // 构造告警 JSON (使用自动获取的本机IP)
    QString hostIp = MainWindow::getHostIp();
    QJsonObject data;
    data["alarm_id"]   = alarmId;
    data["alarm_type"] = alarmType;
    data["timestamp"]  = QDateTime::currentDateTime().toMSecsSinceEpoch();
    data["video_url"]  = QString("http://%1:%2/%3").arg(
        hostIp).arg(Config::HTTP_PORT).arg(baseName + ".mp4");
    data["image_url"]  = QString("http://%1:%2/%3").arg(
        hostIp).arg(Config::HTTP_PORT).arg(baseName + ".jpg");

    QJsonObject root;
    root["type"] = "alarm";
    root["data"] = data;

    QString alertJson = QString::fromUtf8(
        QJsonDocument(root).toJson(QJsonDocument::Compact));

    emit alertSaved(cameraId_, videoPath, imagePath, alertJson);
}


// ============================================================
// MJPEG 推流
// ============================================================
void MainWindow::startMjpegServer() {
    auto& cfg = RuntimeConfig::instance();
    mjpegServer_ = new QTcpServer(this);
    if (!mjpegServer_->listen(QHostAddress::Any, (quint16)cfg.streamPort())) {
        log(QString::fromUtf8("MJPEG"),
            QString::fromUtf8("服务启动失败, 端口: %1").arg(cfg.streamPort()));
        return;
    }
    connect(mjpegServer_, &QTcpServer::newConnection, this, [this]() {
        auto* socket = mjpegServer_->nextPendingConnection();
        if (!socket) return;
        mjpegClients_.append(socket);
        connect(socket, &QTcpSocket::disconnected, this, [this, socket]() {
            mjpegClients_.removeAll(socket);
            socket->deleteLater();
        });
    });
    log(QString::fromUtf8("MJPEG"),
        QString::fromUtf8("服务已启动: http://%1:%2/stream")
            .arg(getHostIp()).arg(cfg.streamPort()));
}

void MainWindow::pushMjpegFrame(const QByteArray& jpegData) {
    if (mjpegClients_.isEmpty()) return;
    QByteArray chunk = QByteArray("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
        + QByteArray::number(jpegData.size()) + QByteArray("\r\n\r\n")
        + jpegData + QByteArray("\r\n");
    auto clients = mjpegClients_;
    for (auto* c : clients) {
        if (c->state() == QAbstractSocket::ConnectedState) c->write(chunk);
    }
}

// ============================================================
// MainWindow
// ============================================================

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , engine_(nullptr)
    , isProcessing_(false)
    , confThreshold_(Config::CONF_THRESHOLD)
    , nmsThreshold_(Config::IOU_THRESHOLD)
{
    ui = new Ui::MainWindow;
    ui->setupUi(this);

    statusMessageLabel_ = new QLabel("就绪", this);
    fpsLabel_  = new QLabel("FPS: --", this);
    timeLabel_ = new QLabel("耗时: --", this);
    wsAddressLabel_ = new QLabel("WebSocket: 未启动", this);
    fpsLabel_->setStyleSheet("margin-right: 15px;");
    timeLabel_->setStyleSheet("margin-right: 15px;");
    wsAddressLabel_->setStyleSheet("margin-right: 15px; color: #337ab7; font-weight: bold;");
    statusBar()->addWidget(statusMessageLabel_, 1);
    statusBar()->addPermanentWidget(wsAddressLabel_);
    statusBar()->addPermanentWidget(fpsLabel_);
    statusBar()->addPermanentWidget(timeLabel_);

    setupConnections();
    statusMessageLabel_->setText("就绪 - 请加载模型后开始检测");
    ui->modelPathEdit->setText(QString::fromStdString(Config::MODEL_PATH));

    QDir().mkpath(QString::fromStdString(Config::OUTPUT_DIR));

    // 加载运行时配置(JSON)
    loadRuntimeConfig();

    startWebSocketServer();
    startMjpegServer();
    if (fs::exists(Config::MODEL_PATH)) onLoadModel();

    ui->batchInferenceCheck->setChecked(Config::USE_BATCH_INFERENCE);

    log("系统", "YOLO11 PPE 检测系统已启动");
    log("配置", QString("模型路径: %1").arg(QString::fromStdString(Config::MODEL_PATH)));
}

MainWindow::~MainWindow() {
    stopAllCameras();
    if (isProcessing_) {
        // 视频模式清理
        isProcessing_ = false;
    }
    for (auto it = pendingAlarms_.begin(); it != pendingAlarms_.end(); ++it) {
        if (it->retryTimer) {
            it->retryTimer->stop();
            delete it->retryTimer;
        }
    }
    pendingAlarms_.clear();
    if (httpServer_) {
        httpServer_->close();
        delete httpServer_;
    }
    for (auto* client : wsClients_) {
        client->close();
    }
    wsClients_.clear();
    if (wsServer_) wsServer_->close();
    for (auto* client : mjpegClients_) client->close();
    mjpegClients_.clear();
    if (mjpegServer_) { mjpegServer_->close(); delete mjpegServer_; }
    delete ui;
}

void MainWindow::setupConnections() {
    connect(ui->openImageBtn,  &QPushButton::clicked, this, &MainWindow::onOpenImage);
    connect(ui->openVideoBtn,  &QPushButton::clicked, this, &MainWindow::onOpenVideo);
    connect(ui->cameraBtn,     &QPushButton::toggled, this, &MainWindow::onOpenCamera);
    connect(ui->addCameraBtn,  &QPushButton::clicked, this, &MainWindow::onAddCamera);
    connect(ui->settingsBtn,    &QPushButton::clicked, this, &MainWindow::onSettings);
    connect(ui->folderBtn,     &QPushButton::clicked, this, &MainWindow::onOpenFolder);
    connect(ui->stopBtn,       &QPushButton::clicked, this, &MainWindow::onStopProcessing);
    connect(ui->browseModelBtn, &QPushButton::clicked, this, &MainWindow::onBrowseModel);
    connect(ui->loadModelBtn,   &QPushButton::clicked, this, &MainWindow::onLoadModel);
    connect(ui->reloadModelBtn, &QPushButton::clicked, this, &MainWindow::onReloadModel);
    connect(ui->confSlider, &QSlider::valueChanged, this, &MainWindow::onConfThresholdChanged);
    connect(ui->nmsSlider,  &QSlider::valueChanged, this, &MainWindow::onNmsThresholdChanged);
    connect(ui->batchInferenceCheck, &QCheckBox::toggled, this, &MainWindow::onBatchInferenceToggled);
    connect(ui->clearLogBtn, &QPushButton::clicked, this, [this]() {
        ui->logTextEdit->clear();
        log("系统", "日志已清空");
    });
    connect(ui->actionOpenImage, &QAction::triggered, this, &MainWindow::onOpenImage);
    connect(ui->actionOpenVideo, &QAction::triggered, this, &MainWindow::onOpenVideo);
    connect(ui->actionOpenCamera, &QAction::triggered, this, [this]() {
        ui->cameraBtn->toggle();
    });
    connect(ui->actionExit, &QAction::triggered, this, &QWidget::close);
    connect(ui->actionLoadModel, &QAction::triggered, this, &MainWindow::onLoadModel);
}

// ============================================================
// WebSocket 服务器
// ============================================================
void MainWindow::startWebSocketServer() {
    wsServer_ = new QWebSocketServer("YOLO11-Alert", QWebSocketServer::NonSecureMode, this);
    if (!wsServer_->listen(QHostAddress::Any, Config::WEBSOCKET_PORT)) {
        wsAddressLabel_->setText("WebSocket: 启动失败");
        log("WebSocket", QString("启动失败, 端口: %1").arg(Config::WEBSOCKET_PORT));
        return;
    }

    connect(wsServer_, &QWebSocketServer::newConnection, this, &MainWindow::onWsClientConnected);

    QString wsAddr = QString("ws://%1:%2")
        .arg(getHostIp())
        .arg(Config::WEBSOCKET_PORT);
    wsAddressLabel_->setText("WebSocket: " + wsAddr);
    log("WebSocket", QString("服务已启动 %1").arg(wsAddr));

    // 启动 HTTP 文件服务器
    startHttpFileServer();
}

// ============================================================
// HTTP 文件服务器(简易): 为告警视频/截图提供下载
// ============================================================
void MainWindow::startHttpFileServer() {
    httpServer_ = new QTcpServer(this);
    if (!httpServer_->listen(QHostAddress::Any, Config::HTTP_PORT)) {
        statusMessageLabel_->setText("HTTP 文件服务启动失败");
        return;
    }

    QDir outputDir(QString::fromStdString(Config::OUTPUT_DIR));
    QString outputPath = outputDir.absolutePath();

    connect(httpServer_, &QTcpServer::newConnection, this, [this, outputPath]() {
        auto* socket = httpServer_->nextPendingConnection();
        if (!socket) return;

        connect(socket, &QTcpSocket::readyRead, this, [socket, outputPath]() {
            // 等待完整的HTTP请求(检查是否以\r\n\r\n结尾)
            if (!socket->canReadLine()) return;
            QByteArray requestData = socket->readAll();
            // 检查请求头是否完整(HTTP头以\r\n\r\n结束)
            if (!requestData.contains("\r\n\r\n")) {
                // 请求头不完整, 等待更多数据
                if (socket->state() == QAbstractSocket::ConnectedState) return;
            }
            QString request = QString::fromUtf8(requestData);
            // 解析 GET /filename.ext HTTP/1.1
            QStringList lines = request.split("\r\n");
            if (lines.isEmpty()) { socket->close(); delete socket; return; }

            QStringList parts = lines[0].split(' ');
            if (parts.size() < 2 || parts[0] != "GET") {
                socket->write("HTTP/1.1 400 Bad Request\r\n\r\n");
                socket->close(); delete socket; return;
            }

            // 获取请求的文件名, 防止路径穿越
            QString fileName = QFileInfo(parts[1]).fileName();
            if (fileName.isEmpty() || fileName.contains("..")) {
                socket->write("HTTP/1.1 404 Not Found\r\n\r\n");
                socket->close(); delete socket; return;
            }

            // 读取文件
            QString filePath = outputPath + "/" + fileName;
            QFile file(filePath);
            if (!file.open(QIODevice::ReadOnly)) {
                socket->write("HTTP/1.1 404 Not Found\r\n\r\n");
                socket->close(); delete socket; return;
            }

            QByteArray data = file.readAll();
            file.close();

            // 根据扩展名设置 Content-Type
            QString ext = QFileInfo(fileName).suffix().toLower();
            QString mime;
            if (ext == "mp4")       mime = "video/mp4";
            else if (ext == "jpg" || ext == "jpeg") mime = "image/jpeg";
            else if (ext == "png")  mime = "image/png";
            else                    mime = "application/octet-stream";

            // 构造 HTTP 响应
            QByteArray header = QString(
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: %1\r\n"
                "Content-Length: %2\r\n"
                "Connection: close\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "\r\n").arg(mime).arg(data.size()).toUtf8();

            socket->write(header + data);
            socket->disconnectFromHost();
        });

        connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
    });

    statusMessageLabel_->setText(
        QString("HTTP 文件服务已启动: http://%1:%2")
            .arg(getHostIp())
            .arg(Config::HTTP_PORT));
}

void MainWindow::onWsClientConnected() {
    auto* socket = wsServer_->nextPendingConnection();
    if (!socket) return;
    wsClients_.append(socket);

    connect(socket, &QWebSocket::textMessageReceived, this, &MainWindow::onWsTextMessage);
    connect(socket, &QWebSocket::disconnected, this, [this, socket]() {
        log("WebSocket", QString("客户端断开连接, 剩余: %1").arg(wsClients_.size() - 1));
        wsClients_.removeAll(socket);
        socket->deleteLater();
    });
    statusMessageLabel_->setText(
        QString("WebSocket 客户端已连接 (%1)").arg(wsClients_.size()));
    log("WebSocket", QString("客户端已连接, 当前连接数: %1").arg(wsClients_.size()));
}

void MainWindow::onWsTextMessage(const QString& message) {
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (doc.isNull()) return;

    QJsonObject obj = doc.object();
    if (obj["type"].toString() != "ack") return;

    QString alarmId = obj["alarm_id"].toString();
    if (alarmId.isEmpty()) return;

    // 找到对应的待确认告警, 停止定时器
    auto it = pendingAlarms_.find(alarmId);
    if (it != pendingAlarms_.end()) {
        if (it->retryTimer) {
            it->retryTimer->stop();
            delete it->retryTimer;
        }
        pendingAlarms_.erase(it);
        statusMessageLabel_->setText(
            QString("告警 %1 已确认").arg(alarmId.left(8)));
        log("WebSocket", QString("收到告警确认: %1").arg(alarmId.left(8)));
    }
}

void MainWindow::retryAlarm(const QString& alarmId) {
    auto it = pendingAlarms_.find(alarmId);
    if (it == pendingAlarms_.end()) return;

    // 超过最大重试次数, 停止重试并清理
    if (++it->retryCount >= MAX_RETRY_COUNT) {
        if (it->retryTimer) {
            it->retryTimer->stop();
            delete it->retryTimer;
        }
        pendingAlarms_.erase(it);
        log("告警", QString("告警 %1 重试超限, 已放弃").arg(alarmId.left(8)));
        return;
    }

    // 重发给所有连接的客户端
    for (auto* client : wsClients_) {
        client->sendTextMessage(it->jsonMessage);
    }
    statusMessageLabel_->setText(
        QString("重发告警 %1 (%2/%3)").arg(alarmId.left(8)).arg(it->retryCount).arg(MAX_RETRY_COUNT));
    log("告警", QString("重发告警: %1 (%2/%3)").arg(alarmId.left(8)).arg(it->retryCount).arg(MAX_RETRY_COUNT));
}

// ============================================================
// 告警处理
// ============================================================
void MainWindow::onAlertSaved(int cameraId, const QString& videoPath, const QString& imagePath,
                               const QString& alertJson) {
    // 解析 alarm_id
    QJsonDocument doc = QJsonDocument::fromJson(alertJson.toUtf8());
    QString alarmId = doc.object()["data"].toObject()["alarm_id"].toString();
    if (alarmId.isEmpty()) return;

    // 广播给所有 WebSocket 客户端
    for (auto* client : wsClients_) {
        client->sendTextMessage(alertJson);
    }

    // 启动 ACK 等待定时器(有最大重试次数)
    auto* timer = new QTimer(this);
    timer->setSingleShot(false);
    connect(timer, &QTimer::timeout, this, [this, alarmId]() {
        retryAlarm(alarmId);
    });

    PendingAlarm pending;
    pending.jsonMessage = alertJson;
    pending.retryTimer = timer;
    pending.retryCount = 0;
    pendingAlarms_[alarmId] = pending;

    timer->start(Config::ACK_TIMEOUT_MS);

    // 状态栏
    QString alarmType = doc.object()["data"].toObject()["alarm_type"].toString();
    statusMessageLabel_->setText(
        QString("[告警] %1 - 视频已保存, 等待确认").arg(alarmType));
    log("告警", QString("发送告警: %1, ID: %2").arg(alarmType, alarmId.left(8)));
    qDebug() << "Alarm sent:" << alertJson;
}

// ============================================================
// 模型管理
// ============================================================
void MainWindow::onBrowseModel() {
    QString filePath = QFileDialog::getOpenFileName(
        this, "选择YOLO11模型文件",
        QString::fromStdString(Config::MODEL_PATH),
        "TensorRT Engine文件 (*.engine);;所有文件 (*)");
    if (!filePath.isEmpty()) ui->modelPathEdit->setText(filePath);
}

void MainWindow::onLoadModel() {
    QString modelPath = ui->modelPathEdit->text().trimmed();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择模型文件路径");
        return;
    }
    if (!fs::exists(modelPath.toStdString())) {
        QMessageBox::warning(this, "错误", "模型文件不存在:\n" + modelPath);
        return;
    }

    ui->loadModelBtn->setEnabled(false);
    ui->modelStatusLabel->setText("加载中...");
    ui->modelStatusLabel->setStyleSheet("color: orange; font-weight: bold;");
    statusMessageLabel_->setText("正在加载模型, 请稍候...");
    QApplication::processEvents();

    try {
        engine_ = std::make_unique<YoloTrtEngine>(modelPath.toStdString());
        ui->modelStatusLabel->setText("✓ 已加载");
        ui->modelStatusLabel->setStyleSheet("color: green; font-weight: bold;");
        statusMessageLabel_->setText("模型加载成功, 可以开始检测");
        log("模型", QString("模型加载成功: %1").arg(modelPath));
        enableControls(true);
        ui->reloadModelBtn->setEnabled(true);
    } catch (const std::exception& e) {
        ui->modelStatusLabel->setText("✗ 加载失败");
        ui->modelStatusLabel->setStyleSheet("color: red; font-weight: bold;");
        statusMessageLabel_->setText("模型加载失败");
        log("模型", QString("模型加载失败: %1").arg(e.what()));
        QMessageBox::critical(this, "模型加载错误", e.what());
        enableControls(false);
    }
    ui->loadModelBtn->setEnabled(true);
}

void MainWindow::onReloadModel() {
    QString modelPath = ui->modelPathEdit->text().trimmed();
    if (modelPath.isEmpty() || !engine_) {
        QMessageBox::warning(this, "警告", "当前没有已加载的模型");
        return;
    }
    if (!fs::exists(modelPath.toStdString())) {
        QMessageBox::warning(this, "警告", "模型文件不存在:\n" + modelPath);
        return;
    }
    log("模型", QString("正在热切换模型: %1").arg(modelPath));
    statusMessageLabel_->setText("正在热切换模型...");
    ui->reloadModelBtn->setEnabled(false);
    stopAllCameras();
    QApplication::processEvents();

    try {
        engine_->reload(modelPath.toStdString());
        ui->modelStatusLabel->setText("✓ 已重载");
        ui->modelStatusLabel->setStyleSheet("color: green; font-weight: bold;");
        statusMessageLabel_->setText("模型热切换成功");
        log("模型", QString("模型热切换成功: %1").arg(modelPath));
    } catch (const std::exception& e) {
        ui->modelStatusLabel->setText("✗ 重载失败");
        ui->modelStatusLabel->setStyleSheet("color: red; font-weight: bold;");
        statusMessageLabel_->setText("模型热切换失败");
        log("错误", QString("模型热切换失败: %1").arg(e.what()));
        QMessageBox::critical(this, "热切换错误", e.what());
    }
    ui->reloadModelBtn->setEnabled(true);
}

// ============================================================
// 推理模式
// ============================================================
void MainWindow::onOpenImage() {
    if (!engine_) { QMessageBox::warning(this, QString::fromUtf8("提示"), QString::fromUtf8("请先加载模型")); return; }
    stopAllCameras();
    QString filePath = QFileDialog::getOpenFileName(
        this, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)");
    if (filePath.isEmpty()) return;
    log("检测", QString("打开图片: %1").arg(filePath));
    statusMessageLabel_->setText("正在推理...");
    QApplication::processEvents();
    processSingleImage(filePath.toStdString());
}

void MainWindow::processSingleImage(const std::string& path) {
    try {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            QMessageBox::warning(this, "错误", "无法读取图像");
            statusMessageLabel_->setText("图像读取失败");
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat processed = Preprocessor::letterbox(img);
        std::vector<float> tensor = Preprocessor::imageToTensor(processed);
        std::vector<Detection> detections;
        engine_->infer(tensor, detections, img.cols, img.rows,
                       confThreshold_, nmsThreshold_);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

        cv::Mat displayImg = img.clone();
        Postprocessor::drawDetections(displayImg, detections);

        cv::Mat rgb;
        cv::cvtColor(displayImg, rgb, cv::COLOR_BGR2RGB);
        QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        updateDisplay(qimg.copy());
        updateDetectionList(detections, elapsedMs);
        statusMessageLabel_->setText(
            QString("推理完成 - 检测到 %1 个目标").arg(detections.size()));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "推理错误", e.what());
        statusMessageLabel_->setText("推理出错");
    }
}

void MainWindow::onOpenVideo() {
    if (!engine_) { QMessageBox::warning(this, QString::fromUtf8("提示"), QString::fromUtf8("请先加载模型")); return; }
    stopAllCameras();
    if (isProcessing_) onStopProcessing();

    QString filePath = QFileDialog::getOpenFileName(
        this, QString::fromUtf8("选择视频文件"), "", QString::fromUtf8("视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"));
    if (filePath.isEmpty()) return;

    log("检测", QString("打开视频: %1").arg(filePath));
    statusMessageLabel_->setText(QString::fromUtf8("正在处理视频..."));
    enableControls(false);
    ui->cameraBtn->setChecked(false);
    ui->stopBtn->setEnabled(true);
    isProcessing_ = true;

    auto* thread = new QThread(this);
    auto* worker = new InferenceWorker(engine_.get(), -1, "video", filePath);
    worker->moveToThread(thread);

    connect(worker, &InferenceWorker::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(worker, &InferenceWorker::alertSaved, this, &MainWindow::onAlertSaved);
    connect(worker, &InferenceWorker::finished, this, [this](int cameraId) {
        // 视频模式结束
        isProcessing_ = false;
        enableControls(true);
        ui->stopBtn->setEnabled(false);
        statusMessageLabel_->setText(QString::fromUtf8("视频处理完成"));
        fpsLabel_->setText("FPS: --");
        log("系统", "视频处理完成");
        // 视频worker不在cameraWorkers_中, 手动清理
        sender()->deleteLater();
    });
    connect(worker, &InferenceWorker::errorOccurred, this, [this](int, const QString& msg) {
        QMessageBox::critical(this, QString::fromUtf8("处理错误"), msg);
    });
    connect(thread, &QThread::started, worker, [this, worker, filePath]() {
        worker->setBatchInference(ui->batchInferenceCheck->isChecked());
        worker->processVideo(filePath, confThreshold_, nmsThreshold_);
    });
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);

    activeDisplayCamera_ = -1;
    thread->start();
}

void MainWindow::onOpenCamera(bool checked) {
    if (checked) {
        if (!engine_) {
            QMessageBox::warning(this, QString::fromUtf8("提示"), QString::fromUtf8("请先加载模型"));
            ui->cameraBtn->setChecked(false);
            return;
        }
        // 如果默认摄像头已在运行, 先停止
        if (cameraWorkers_.contains(0)) stopCamera(0);

        log("检测", "启动默认摄像头");
        statusMessageLabel_->setText(QString::fromUtf8("正在启动摄像头..."));
        ui->cameraBtn->setText(QString::fromUtf8("关闭摄像头"));
        ui->cameraStatusLabel->setStyleSheet("font-size: 20px; font-weight: bold; padding: 0 12px; color: green;");
        ui->cameraStatusLabel->setText(QString::fromUtf8("● 已开启"));
        activeDisplayCamera_ = 0;

        auto* thread = new QThread(this);
        auto* worker = new InferenceWorker(engine_.get(), 0, "camera_0");
        worker->moveToThread(thread);

        connect(worker, &InferenceWorker::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(worker, &InferenceWorker::alertSaved, this, &MainWindow::onAlertSaved);
        connect(worker, &InferenceWorker::finished, this, &MainWindow::onWorkerFinished);
        connect(worker, &InferenceWorker::errorOccurred, this, &MainWindow::onWorkerError);
        connect(thread, &QThread::started, worker, [this, worker]() {
            worker->setBatchInference(ui->batchInferenceCheck->isChecked());
            worker->processCamera(confThreshold_, nmsThreshold_);
        });
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);

        CameraWorker cw{thread, worker, nullptr};
        cameraWorkers_[0] = cw;
        thread->start();
    } else {
        stopCamera(0);
    }
}

void MainWindow::onAddCamera() {
    if (!engine_) {
        QMessageBox::warning(this, QString::fromUtf8("提示"), QString::fromUtf8("请先加载模型"));
        return;
    }

    // 弹出对话框: 输入摄像头设备ID或RTSP地址
    bool ok = false;
    QString source = QInputDialog::getText(this,
        QString::fromUtf8("添加摄像头"),
        QString::fromUtf8("输入摄像头设备ID(0,1,2...)或RTSP地址:"),
        QLineEdit::Normal, "", &ok);
    if (!ok || source.isEmpty()) return;

    int camId = nextCameraId_++;
    QString camName = QString("camera_%1").arg(camId);

    log("检测", QString("添加摄像头 %1: %2").arg(camId).arg(source));
    activeDisplayCamera_ = camId;

    auto* thread = new QThread(this);
    auto* worker = new InferenceWorker(engine_.get(), camId, camName, source);
    worker->moveToThread(thread);

    connect(worker, &InferenceWorker::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(worker, &InferenceWorker::alertSaved, this, &MainWindow::onAlertSaved);
    connect(worker, &InferenceWorker::finished, this, &MainWindow::onWorkerFinished);
    connect(worker, &InferenceWorker::errorOccurred, this, &MainWindow::onWorkerError);
    connect(thread, &QThread::started, worker, [this, worker]() {
        worker->setBatchInference(ui->batchInferenceCheck->isChecked());
        worker->processSource(confThreshold_, nmsThreshold_);
    });
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);

    CameraWorker cw{thread, worker, nullptr};
    cameraWorkers_[camId] = cw;
    thread->start();

    // 更新摄像头状态显示
    ui->cameraStatusLabel->setStyleSheet("font-size: 20px; font-weight: bold; padding: 0 12px; color: green;");
    ui->cameraStatusLabel->setText(QString::fromUtf8("● %1路运行").arg(cameraWorkers_.size()));
    ui->stopBtn->setEnabled(true);
}

void MainWindow::onRemoveCamera(int cameraId) {
    stopCamera(cameraId);
}

void MainWindow::stopCamera(int cameraId) {
    auto it = cameraWorkers_.find(cameraId);
    if (it == cameraWorkers_.end()) return;

    if (it->worker) it->worker->stop();
    if (it->thread) {
        if (!it->thread->wait(8000)) {
            log("系统", QString("摄像头 %1 线程未响应，强制清理").arg(cameraId));
        }
    }

    // 如果是默认摄像头, 重置UI
    if (cameraId == 0) {
        ui->cameraBtn->setChecked(false);
        ui->cameraBtn->setText(QString::fromUtf8("开启摄像头"));
    }

    cameraWorkers_.erase(it);

    // 强制释放摄像头资源
    cv::VideoCapture forceRelease(cameraId, cv::CAP_DSHOW);
    if (forceRelease.isOpened()) forceRelease.release();

    if (cameraWorkers_.isEmpty()) {
        ui->cameraStatusLabel->setStyleSheet("font-size: 20px; font-weight: bold; padding: 0 12px;");
        ui->cameraStatusLabel->setText(QString::fromUtf8("⏹ 未开启"));
        ui->stopBtn->setEnabled(false);
        fpsLabel_->setText("FPS: --");
    } else {
        ui->cameraStatusLabel->setText(QString::fromUtf8("● %1路运行").arg(cameraWorkers_.size()));
    }

    log("系统", QString("摄像头 %1 已停止").arg(cameraId));
}

void MainWindow::stopAllCameras() {
    // 复制key列表避免迭代时修改
    auto ids = cameraWorkers_.keys();
    for (int id : ids) {
        stopCamera(id);
    }
}

void MainWindow::onOpenFolder() {
    if (!engine_) { QMessageBox::warning(this, QString::fromUtf8("提示"), QString::fromUtf8("请先加载模型")); return; }
    stopAllCameras();

    QString dirPath = QFileDialog::getExistingDirectory(
        this, "选择图片文件夹", "", QFileDialog::ShowDirsOnly);
    if (dirPath.isEmpty()) return;

    log("检测", QString("批量处理文件夹: %1").arg(dirPath));

    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    int total = 0, succ = 0;
    statusMessageLabel_->setText("正在批量处理...");
    QApplication::processEvents();

    try {
        for (const auto& entry : fs::directory_iterator(dirPath.toStdString())) {
            if (!fs::is_regular_file(entry)) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                try {
                    cv::Mat img = cv::imread(entry.path().string());
                    if (img.empty()) continue;
                    cv::Mat processed = Preprocessor::letterbox(img);
                    std::vector<float> tensor = Preprocessor::imageToTensor(processed);
                    std::vector<Detection> detections;
                    engine_->infer(tensor, detections, img.cols, img.rows,
                                   confThreshold_, nmsThreshold_);
                    Postprocessor::drawDetections(img, detections);
                    cv::imwrite(Config::OUTPUT_DIR + "/result_" +
                                entry.path().filename().string(), img);
                    succ++;
                } catch (...) {}
                total++;
                statusMessageLabel_->setText(
                    QString("批量处理: %1/%2").arg(succ).arg(total));
                QApplication::processEvents();
            }
        }
        statusMessageLabel_->setText(
            QString("批量处理完成 - 成功处理 %1/%2 张图片").arg(succ).arg(total));
        QMessageBox::information(this, "批量处理完成",
            QString("共扫描 %1 张\n成功处理 %2 张\n结果保存至: %3")
                .arg(total).arg(succ).arg(QString::fromStdString(Config::OUTPUT_DIR)));
    } catch (...) {
        QMessageBox::critical(this, "批量处理错误", "处理过程中发生错误");
    }
}

void MainWindow::onStopProcessing() {
    stopAllCameras();
    isProcessing_ = false;
    enableControls(true);
    ui->stopBtn->setEnabled(false);
    fpsLabel_->setText("FPS: --");
    statusMessageLabel_->setText(QString::fromUtf8("已停止"));
    log("系统", "所有处理已停止");
}

void MainWindow::onFrameProcessed(int cameraId, QImage image, std::vector<Detection> detections, double elapsedMs) {
    // MJPEG推流: 每帧都推
    QByteArray jpegData;
    {
        QBuffer buf(&jpegData);
        buf.open(QIODevice::WriteOnly);
        image.save(&buf, "JPEG", 60);
    }
    pushMjpegFrame(jpegData);

    // 只更新当前活跃显示的摄像头画面
    if (cameraId == activeDisplayCamera_) {
        updateDisplay(image);
        updateDetectionList(detections, elapsedMs);
    }
    if (elapsedMs > 0 && cameraId == activeDisplayCamera_) {
        double fps = 1000.0 / elapsedMs;
        fpsLabel_->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    }
    // 每5帧输出一次检测详情日志
    static int logFrameCount = 0;
    if (++logFrameCount % 5 == 0) {
        if (detections.empty()) {
            log("检测", "未检测到目标");
        } else {
            // 统计各类别数量
            std::map<int, int> classCounts;
            for (const auto& det : detections) {
                classCounts[det.class_id]++;
            }
            QString detail;
            for (const auto& [cid, cnt] : classCounts) {
                if (!detail.isEmpty()) detail += ", ";
                detail += QString("%1×%2").arg(cnt).arg(QString::fromStdString(Config::CLASS_NAMES[cid]));
            }
            double fps = (elapsedMs > 0) ? 1000.0 / elapsedMs : 0;
            log("检测", QString("%1 | %2 | %3ms, FPS:%4")
                .arg(detections.size()).arg(detail)
                .arg(elapsedMs, 0, 'f', 1).arg(fps, 0, 'f', 1));
        }
    }
}

void MainWindow::onWorkerFinished(int cameraId) {
    auto it = cameraWorkers_.find(cameraId);
    if (it != cameraWorkers_.end()) {
        if (it->thread) {
            it->thread->quit();
            it->thread->wait();
        }
        // 如果是默认摄像头, 重置UI
        if (cameraId == 0) {
            ui->cameraBtn->setChecked(false);
            ui->cameraBtn->setText(QString::fromUtf8("开启摄像头"));
            ui->cameraStatusLabel->setStyleSheet("font-size: 20px; font-weight: bold; padding: 0 12px;");
            ui->cameraStatusLabel->setText(QString::fromUtf8("⏹ 未开启"));
        }
        cameraWorkers_.erase(it);
        log("系统", QString("摄像头 %1 处理完成").arg(cameraId));
    }

    // 如果没有任何活跃worker, 重置UI
    if (cameraWorkers_.isEmpty()) {
        isProcessing_ = false;
        enableControls(true);
        ui->stopBtn->setEnabled(false);
        fpsLabel_->setText("FPS: --");
        statusMessageLabel_->setText("处理完成");
    }
}

void MainWindow::onWorkerError(int cameraId, const QString& message) {
    log("错误", QString("[摄像头%1] %2").arg(cameraId).arg(message));
    if (cameraId == 0) {
        ui->cameraBtn->setChecked(false);
    }
    onWorkerFinished(cameraId);
}

void MainWindow::onConfThresholdChanged(int value) {
    confThreshold_ = value / 100.0f;
    updateThresholdLabels();
}

void MainWindow::onNmsThresholdChanged(int value) {
    nmsThreshold_ = value / 100.0f;
    updateThresholdLabels();
}

void MainWindow::updateThresholdLabels() {
    ui->confValueLabel->setText(QString::number(confThreshold_, 'f', 2));
    ui->nmsValueLabel->setText(QString::number(nmsThreshold_, 'f', 2));
}

void MainWindow::updateDisplay(const QImage& image) {
    QPixmap pixmap = QPixmap::fromImage(image);
    QSize labelSize = ui->displayLabel->size();
    // 实时视频用FastTransformation，避免SmoothTransformation卡死主线程
    pixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::FastTransformation);
    ui->displayLabel->setPixmap(pixmap);
}

void MainWindow::updateDetectionList(const std::vector<Detection>& detections, double elapsedMs) {
    // 增量更新：只在检测数量变化时才重建列表，避免每帧clear+rebuild
    int newCount = static_cast<int>(detections.size());
    int oldCount = ui->resultListWidget->count();
    bool needRebuild = (newCount != oldCount);

    // 如果数量相同，检查类别是否变化
    if (!needRebuild && newCount > 0) {
        for (int i = 0; i < newCount && i < oldCount; ++i) {
            auto* item = ui->resultListWidget->item(i);
            const auto& name = Config::CLASS_NAMES[detections[i].class_id];
            QString text = QString("%1  %2")
                .arg(QString::fromStdString(name), -12).arg(detections[i].conf, 0, 'f', 3);
            if (item->text() != text) { needRebuild = true; break; }
        }
    }

    if (needRebuild) {
        ui->resultListWidget->clear();
        if (detections.empty()) {
            ui->resultListWidget->addItem("未检测到目标");
        } else {
            std::vector<Detection> sorted = detections;
            std::sort(sorted.begin(), sorted.end(),
                      [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

            for (const auto& det : sorted) {
                const auto& name = Config::CLASS_NAMES[det.class_id];
                QString text = QString("%1  %2")
                    .arg(QString::fromStdString(name), -12).arg(det.conf, 0, 'f', 3);
                auto* item = new QListWidgetItem(text, ui->resultListWidget);
                if (name.find("no_") == 0)
                    item->setForeground(QColor("#d9534f"));
                else if (name == "Person" || name == "none")
                    item->setForeground(QColor("#888888"));
                else
                    item->setForeground(QColor("#5cb85c"));
            }
        }
    } else {
        // 数量相同，只更新置信度数值
        std::vector<Detection> sorted = detections;
        std::sort(sorted.begin(), sorted.end(),
                  [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
            auto* item = ui->resultListWidget->item(i);
            const auto& name = Config::CLASS_NAMES[sorted[i].class_id];
            QString text = QString("%1  %2")
                .arg(QString::fromStdString(name), -12).arg(sorted[i].conf, 0, 'f', 3);
            item->setText(text);
            if (name.find("no_") == 0)
                item->setForeground(QColor("#d9534f"));
            else if (name == "Person" || name == "none")
                item->setForeground(QColor("#888888"));
            else
                item->setForeground(QColor("#5cb85c"));
        }
    }

    ui->totalCountLabel->setText(QString("目标总数: %1").arg(detections.size()));
    timeLabel_->setText(QString("耗时: %1ms").arg(elapsedMs, 0, 'f', 1));
}

void MainWindow::enableControls(bool enabled) {
    ui->openImageBtn->setEnabled(enabled);
    ui->openVideoBtn->setEnabled(enabled);
    ui->cameraBtn->setEnabled(enabled);
    ui->folderBtn->setEnabled(enabled);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    stopAllCameras();
    if (isProcessing_) isProcessing_ = false;
    event->accept();
}

// ============================================================
// 批量推理开关
// ============================================================
void MainWindow::onBatchInferenceToggled(bool checked) {
    if (checked) {
        if (Config::BATCH_SIZE <= 1) {
            ui->batchInferenceCheck->setChecked(false);
            log(QString::fromUtf8("配置"), QString::fromUtf8("当前BATCH_SIZE=1, 无法启用批量推理"));
            return;
        }
        // 设置所有活跃worker
        for (auto& cw : cameraWorkers_) {
            if (cw.worker) cw.worker->setBatchInference(true);
        }
        statusMessageLabel_->setText(QString::fromUtf8("批量推理已启用 (batch=%1)").arg(Config::BATCH_SIZE));
        log(QString::fromUtf8("配置"), QString::fromUtf8("批量推理已启用 (batch=%1)").arg(Config::BATCH_SIZE));
    } else {
        for (auto& cw : cameraWorkers_) {
            if (cw.worker) cw.worker->setBatchInference(false);
        }
        statusMessageLabel_->setText(QString::fromUtf8("批量推理已禁用, 使用单帧推理"));
        log(QString::fromUtf8("配置"), QString::fromUtf8("批量推理已禁用"));
    }
}

// ============================================================
// 日志输出
// ============================================================
QString MainWindow::currentTimestamp() {
    return QDateTime::currentDateTime().toString("hh:mm:ss");
}

void MainWindow::log(const QString& category, const QString& message) {
    QString timestamp = currentTimestamp();
    QString formatted = QString("[%1][%2] %3").arg(timestamp, category, message);
    ui->logTextEdit->append(formatted);
    // 自动滚动到底部
    QTextCursor cursor = ui->logTextEdit->textCursor();
    cursor.movePosition(QTextCursor::End);
    ui->logTextEdit->setTextCursor(cursor);
}

// ============================================================
// 运行时配置
// ============================================================
void MainWindow::loadRuntimeConfig() {
    auto& cfg = RuntimeConfig::instance();
    QString cfgPath = QDir::currentPath() + "/config.json";
    if (QFile::exists(cfgPath)) {
        if (cfg.loadFromFile(cfgPath)) {
            log(QString::fromUtf8("配置"), QString::fromUtf8("已加载运行时配置: %1").arg(cfgPath));
        } else {
            log(QString::fromUtf8("配置"), QString::fromUtf8("配置文件解析失败, 使用默认值"));
        }
    } else {
        // 首次运行, 生成默认配置文件
        cfg.saveToFile(cfgPath);
        log(QString::fromUtf8("配置"), QString::fromUtf8("已生成默认配置: %1").arg(cfgPath));
    }

    // 用运行时配置覆盖UI初始值
    confThreshold_ = cfg.confThreshold();
    nmsThreshold_  = cfg.iouThreshold();
    ui->modelPathEdit->setText(cfg.modelPath());
    ui->confSlider->setValue(static_cast<int>(confThreshold_ * 100));
    ui->nmsSlider->setValue(static_cast<int>(nmsThreshold_ * 100));
}

void MainWindow::saveRuntimeConfig() {
    auto& cfg = RuntimeConfig::instance();
    cfg.setConfThreshold(confThreshold_);
    cfg.setIouThreshold(nmsThreshold_);
    cfg.setModelPath(ui->modelPathEdit->text());
    QString cfgPath = QDir::currentPath() + "/config.json";
    cfg.saveToFile(cfgPath);
    log(QString::fromUtf8("配置"), QString::fromUtf8("配置已保存: %1").arg(cfgPath));
}

void MainWindow::onSettings() {
    auto& cfg = RuntimeConfig::instance();

    auto* dlg = new QDialog(this);
    dlg->setWindowTitle(QString::fromUtf8("运行时设置"));
    dlg->setMinimumWidth(480);
    auto* layout = new QFormLayout(dlg);

    // 阈值
    auto* confSpin = new QDoubleSpinBox(dlg);
    confSpin->setRange(0.01, 0.99); confSpin->setSingleStep(0.05);
    confSpin->setDecimals(2); confSpin->setValue(confThreshold_);
    layout->addRow(QString::fromUtf8("置信度阈值:"), confSpin);

    auto* nmsSpin = new QDoubleSpinBox(dlg);
    nmsSpin->setRange(0.01, 0.99); nmsSpin->setSingleStep(0.05);
    nmsSpin->setDecimals(2); nmsSpin->setValue(nmsThreshold_);
    layout->addRow(QString::fromUtf8("NMS IOU阈值:"), nmsSpin);

    // 端口
    auto* wsPortSpin = new QSpinBox(dlg);
    wsPortSpin->setRange(1024, 65535); wsPortSpin->setValue(cfg.websocketPort());
    layout->addRow(QString::fromUtf8("WebSocket端口:"), wsPortSpin);

    auto* httpPortSpin = new QSpinBox(dlg);
    httpPortSpin->setRange(1024, 65535); httpPortSpin->setValue(cfg.httpPort());
    layout->addRow(QString::fromUtf8("HTTP端口:"), httpPortSpin);

    auto* streamPortSpin = new QSpinBox(dlg);
    streamPortSpin->setRange(1024, 65535); streamPortSpin->setValue(cfg.streamPort());
    layout->addRow(QString::fromUtf8("MJPEG流端口:"), streamPortSpin);

    // 告警参数
    auto* ackSpin = new QSpinBox(dlg);
    ackSpin->setRange(1000, 60000); ackSpin->setSingleStep(1000); ackSpin->setValue(cfg.ackTimeoutMs());
    ackSpin->setSuffix(" ms");
    layout->addRow(QString::fromUtf8("ACK超时:"), ackSpin);

    auto* cooldownSpin = new QSpinBox(dlg);
    cooldownSpin->setRange(1000, 60000); cooldownSpin->setSingleStep(1000); cooldownSpin->setValue(cfg.alertCooldownMs());
    cooldownSpin->setSuffix(" ms");
    layout->addRow(QString::fromUtf8("告警冷却:"), cooldownSpin);

    auto* ringSpin = new QSpinBox(dlg);
    ringSpin->setRange(10, 300); ringSpin->setValue(cfg.ringBufferFrames());
    layout->addRow(QString::fromUtf8("环形缓冲帧数:"), ringSpin);

    // 路径
    auto* modelEdit = new QLineEdit(cfg.modelPath(), dlg);
    layout->addRow(QString::fromUtf8("模型路径:"), modelEdit);

    auto* outputEdit = new QLineEdit(cfg.outputDir(), dlg);
    layout->addRow(QString::fromUtf8("输出目录:"), outputEdit);

    auto* recordEdit = new QLineEdit(cfg.recordDir(), dlg);
    layout->addRow(QString::fromUtf8("录像目录:"), recordEdit);

    // 按钮
    auto* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, dlg);
    layout->addRow(btns);

    connect(btns, &QDialogButtonBox::accepted, dlg, &QDialog::accept);
    connect(btns, &QDialogButtonBox::rejected, dlg, &QDialog::reject);

    if (dlg->exec() == QDialog::Accepted) {
        // 应用设置
        confThreshold_ = (float)confSpin->value();
        nmsThreshold_  = (float)nmsSpin->value();
        ui->confSlider->setValue(static_cast<int>(confThreshold_ * 100));
        ui->nmsSlider->setValue(static_cast<int>(nmsThreshold_ * 100));

        cfg.setConfThreshold(confThreshold_);
        cfg.setIouThreshold(nmsThreshold_);
        cfg.setWebsocketPort(wsPortSpin->value());
        cfg.setHttpPort(httpPortSpin->value());
        cfg.setStreamPort(streamPortSpin->value());
        cfg.setAckTimeoutMs(ackSpin->value());
        cfg.setAlertCooldownMs(cooldownSpin->value());
        cfg.setRingBufferFrames(ringSpin->value());
        cfg.setModelPath(modelEdit->text());
        cfg.setOutputDir(outputEdit->text());
        cfg.setRecordDir(recordEdit->text());

        // 保存到JSON
        saveRuntimeConfig();

        log(QString::fromUtf8("配置"), QString::fromUtf8("运行时设置已更新(端口变更需重启生效)"));
    }

    delete dlg;
}

// ============================================================
// 工具方法
// ============================================================
QString MainWindow::getHostIp() {
    // 优先获取非回环的IPv4地址
    const auto interfaces = QNetworkInterface::allInterfaces();
    for (const auto& iface : interfaces) {
        if (iface.flags() & QNetworkInterface::IsLoopBack) continue;
        if (!(iface.flags() & QNetworkInterface::IsUp)) continue;
        if (!(iface.flags() & QNetworkInterface::IsRunning)) continue;
        for (const auto& addr : iface.addressEntries()) {
            if (addr.ip().protocol() == QAbstractSocket::IPv4Protocol &&
                addr.ip() != QHostAddress::LocalHost) {
                return addr.ip().toString();
            }
        }
    }
    // 回退到配置文件中的IP
    return QString::fromStdString(Config::HOST_IP);
}
