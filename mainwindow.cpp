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
#include <QDialog>
#include <QDialogButtonBox>

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
        // 设置读取超时，避免长时间阻塞
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
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
    // 正常退出时释放资源
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
        
        // 发送HTTP响应头
        QByteArray header = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Connection: keep-alive\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        socket->write(header);
        socket->flush();
        
        mjpegClients_.append(socket);
        connect(socket, &QTcpSocket::disconnected, this, [this, socket]() {
            mjpegClients_.removeAll(socket);
            socket->deleteLater();
            log(QString::fromUtf8("MJPEG"), "客户端已断开");
        });
        log(QString::fromUtf8("MJPEG"), "新客户端已连接");
    });
    log(QString::fromUtf8("MJPEG"),
        QString::fromUtf8("服务已启动: http://%1:%2/stream")
            .arg(QString::fromStdString(Config::HOST_IP)).arg(cfg.streamPort()));
}

void MainWindow::pushMjpegFrame(const QByteArray& jpegData) {
    if (mjpegClients_.isEmpty()) return;

    // MJPEG推帧频率控制在约30fps (每帧间隔33ms)
    static auto lastFrameTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();

    // 如果距离上一帧不足33ms，跳过此帧
    if (elapsed < 33) return;

    lastFrameTime = now;

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
    // 录制按钮
    connect(ui->startRecordBtn, &QPushButton::clicked, this, &MainWindow::onStartRecording);
    connect(ui->stopRecordBtn, &QPushButton::clicked, this, &MainWindow::onStopRecording);
    connect(ui->viewRecordBtn, &QPushButton::clicked, this, &MainWindow::onViewRecordings);
    connect(ui->clearRecordBtn, &QPushButton::clicked, this, &MainWindow::onClearOldRecordings);
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
    //    .arg(getHostIp())
        .arg(QString::fromStdString(Config::HOST_IP))
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
            .arg(QString::fromStdString(Config::HOST_IP))
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
    QString type = obj["type"].toString();

    // 处理 ping 心跳
    if (type == "ping") {
        QJsonObject pong;
        pong["type"] = "pong";
        QJsonDocument pongDoc(pong);

        // 获取发送消息的客户端
        auto* client = qobject_cast<QWebSocket*>(sender());
        if (client) {
            client->sendTextMessage(pongDoc.toJson(QJsonDocument::Compact));
        }
        log("WebSocket", "收到 ping, 回复 pong");
        return;
    }

    // 处理 sync_request 同步请求
    if (type == "sync_request") {
        QString lastAlarmId = obj["last_alarm_id"].toString();
        handleSyncRequest(lastAlarmId);
        return;
    }

    // 处理 get_streams 获取摄像头列表
    if (type == "get_streams") {
        handleGetStreams();
        return;
    }

    // 处理 set_fence 围栏设置
    if (type == "set_fence") {
        QString streamId = obj["stream_id"].toString();
        QJsonObject fence = obj["fence"].toObject();
        handleSetFence(streamId, fence);
        return;
    }

    // 处理 view_stream 查看实时视频
    if (type == "view_stream") {
        QString streamId = obj["stream_id"].toString();
        handleViewStream(streamId);
        return;
    }

    // 处理 ack 告警确认
    if (type != "ack") return;

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
// WebSocket 消息处理 - 同步请求
// ============================================================
void MainWindow::handleSyncRequest(const QString& lastAlarmId) {
    log("WebSocket", QString("收到同步请求, last_alarm_id: %1").arg(lastAlarmId));

    // 找出 last_alarm_id 之后且未确认的告警，逐条推送
    bool foundLast = lastAlarmId.isEmpty();  // 如果为空，推送所有

    // 遍历所有待确认的告警
    for (auto it = pendingAlarms_.begin(); it != pendingAlarms_.end(); ++it) {
        const QString& alarmId = it.key();

        // 如果找到了 last_alarm_id，从这里之后开始推送
        if (!foundLast && alarmId == lastAlarmId) {
            foundLast = true;
            continue;  // 跳过 last_alarm_id 本身
        }

        // 推送 last_alarm_id 之后的告警
        if (foundLast) {
            for (auto* client : wsClients_) {
                client->sendTextMessage(it->jsonMessage);
            }
            log("WebSocket", QString("同步推送告警: %1").arg(alarmId.left(8)));
        }
    }

    if (lastAlarmId.isEmpty()) {
        log("WebSocket", "同步完成: 推送所有待确认告警");
    } else {
        log("WebSocket", QString("同步完成: last_alarm_id=%1").arg(lastAlarmId.left(8)));
    }
}

// ============================================================
// WebSocket 消息处理 - 获取摄像头列表
// ============================================================
void MainWindow::handleGetStreams() {
    QJsonArray streamsArray;

    // 遍历所有活跃的摄像头worker
    for (auto it = cameraWorkers_.begin(); it != cameraWorkers_.end(); ++it) {
        int camId = it.key();
        const auto& worker = it.value();

        QJsonObject stream;
        stream["stream_id"] = QString::number(camId);
        stream["name"] = worker.worker ? worker.worker->cameraName() : QString("摄像头%1").arg(camId);
        // URL 使用MJPEG流地址
        stream["url"] = QString("http://%1:%2/stream")
            .arg(QString::fromStdString(Config::HOST_IP))
            .arg(RuntimeConfig::instance().streamPort());

        streamsArray.append(stream);
    }

    QJsonObject response;
    response["type"] = "streams_list";
    response["data"] = streamsArray;

    QJsonDocument doc(response);
    QString jsonStr = doc.toJson(QJsonDocument::Compact);

    // 发送给请求的客户端
    auto* client = qobject_cast<QWebSocket*>(sender());
    if (client) {
        client->sendTextMessage(jsonStr);
    }

    log("WebSocket", QString("响应摄像头列表: %1 个流").arg(streamsArray.size()));
}

// ============================================================
// WebSocket 消息处理 - 设置围栏
// ============================================================
void MainWindow::handleSetFence(const QString& streamId, const QJsonObject& fence) {
    float x1 = fence["x1"].toDouble(0.0);
    float y1 = fence["y1"].toDouble(0.0);
    float x2 = fence["x2"].toDouble(1.0);
    float y2 = fence["y2"].toDouble(1.0);

    FenceRegion region;
    region.x1 = x1;
    region.y1 = y1;
    region.x2 = x2;
    region.y2 = y2;

    fenceRegions_[streamId] = region;

    log("WebSocket", QString("设置围栏: stream_id=%1, 区域=(%.2f,%.2f,%.2f,%.2f)")
        .arg(streamId).arg(x1).arg(y1).arg(x2).arg(y2));

    // 响应确认
    QJsonObject response;
    response["type"] = "fence_set";
    response["stream_id"] = streamId;
    response["status"] = "success";

    QJsonDocument doc(response);
    auto* client = qobject_cast<QWebSocket*>(sender());
    if (client) {
        client->sendTextMessage(doc.toJson(QJsonDocument::Compact));
    }
}

// ============================================================
// WebSocket 消息处理 - 查看实时视频流
// ============================================================
void MainWindow::handleViewStream(const QString& streamId) {
    // 检查摄像头是否存在
    int camId = streamId.toInt();
    auto it = cameraWorkers_.find(camId);
    if (it == cameraWorkers_.end()) {
        QJsonObject error;
        error["type"] = "stream_error";
        error["stream_id"] = streamId;
        error["message"] = "摄像头不存在";
        
        auto* client = qobject_cast<QWebSocket*>(sender());
        if (client) {
            client->sendTextMessage(QJsonDocument(error).toJson(QJsonDocument::Compact));
        }
        return;
    }

    // 返回MJPEG流URL
    QString streamUrl = QString("http://%1:%2/stream")
        .arg(QString::fromStdString(Config::HOST_IP))
        .arg(RuntimeConfig::instance().streamPort());

    QJsonObject response;
    response["type"] = "stream_url";
    response["stream_id"] = streamId;
    response["url"] = streamUrl;
    response["message"] = "请在浏览器中打开此URL查看实时视频";

    auto* client = qobject_cast<QWebSocket*>(sender());
    if (client) {
        client->sendTextMessage(QJsonDocument(response).toJson(QJsonDocument::Compact));
    }

    log("WebSocket", QString("请求查看摄像头%1的实时视频").arg(streamId));
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

        // 自动开始录制
        QTimer::singleShot(500, this, [this]() {
            onStartRecording();
        });
    } else {
        // 停止录制（如果正在录制）
        auto recIt = cameraRecordings_.find(0);
        if (recIt != cameraRecordings_.end() && recIt.value()->isRecording) {
            onStopRecording();
        }
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

    // 自动开始录制
    QTimer::singleShot(500, this, [this, camId]() {
        activeDisplayCamera_ = camId;
        onStartRecording();
    });
}

void MainWindow::onRemoveCamera(int cameraId) {
    stopCamera(cameraId);
}

void MainWindow::stopCamera(int cameraId) {
    auto it = cameraWorkers_.find(cameraId);
    if (it == cameraWorkers_.end()) return;

    log("系统", QString("正在停止摄像头 %1...").arg(cameraId));

    // 0. 如果该摄像头正在录制，先停止录制
    auto recIt = cameraRecordings_.find(cameraId);
    if (recIt != cameraRecordings_.end() && recIt.value()->isRecording) {
        log("录像", QString("摄像头%1正在录制，自动停止录制").arg(cameraId));
        activeDisplayCamera_ = cameraId;
        onStopRecording();
    }

    // 1. 设置停止标志
    if (it->worker) it->worker->stop();

    // 2. 等待线程退出（缩短超时时间到2秒）
    if (it->thread) {
        if (!it->thread->wait(2000)) {
            log("系统", QString("摄像头 %1 线程未响应，强制终止").arg(cameraId));
            // 不再使用强制释放，避免阻塞
            it->thread->terminate();
            it->thread->wait(500);
        }
    }

    // 3. 清理资源
    cameraWorkers_.erase(it);

    // 4. 更新UI状态
    if (cameraId == 0) {
        ui->cameraBtn->setChecked(false);
        ui->cameraBtn->setText(QString::fromUtf8("开启摄像头"));
    }

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
    // 录像: 写帧到录像文件
    auto recIt = cameraRecordings_.find(cameraId);
    if (recIt != cameraRecordings_.end()) {
        CameraRecording* rec = recIt.value();
        if (rec && rec->isRecording && rec->writer && rec->writer->isOpened()) {
            // QImage 转 cv::Mat
            QImage convImg = image.convertToFormat(QImage::Format_RGB888);
            cv::Mat mat(convImg.height(), convImg.width(), CV_8UC3, const_cast<uchar*>(convImg.constBits()), convImg.bytesPerLine());
            cv::Mat bgr;
            cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
            // 缩放到1080p
            cv::Mat resized;
            cv::resize(bgr, resized, cv::Size(1920, 1080));
            rec->writer->write(resized);
        }
    }

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
    // 每5帧输出一次检测详情日志（只在检测到目标时输出）
    static int logFrameCount = 0;
    if (++logFrameCount % 5 == 0) {
        if (!detections.empty()) {
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
    // 停止所有录像
    for (auto it = cameraRecordings_.begin(); it != cameraRecordings_.end(); ++it) {
        auto rec = it.value();
        if (rec) {
            if (rec->isRecording && rec->writer && rec->writer->isOpened()) {
                rec->writer->release();
            }
            if (rec->writer) delete rec->writer;
            delete rec;  // ← 修复: 释放 CameraRecording 对象本身
        }
    }
    cameraRecordings_.clear();
    if (isProcessing_) isProcessing_ = false;
    event->accept();
}

// ============================================================
// 视频录制功能
// ============================================================
QString MainWindow::getRecordDir(int cameraId) {
    QString baseDir = QDir::cleanPath(QDir::currentPath() + "/" + QString::fromStdString(Config::RECORD_DIR));
    QDateTime now = QDateTime::currentDateTime();
    QString dateDir = now.toString("yyyyMMdd");
    QString cameraDir = QString("camera_%1").arg(cameraId);
    QString fullPath = baseDir + "/" + dateDir + "/" + cameraDir;
    QDir().mkpath(fullPath);
    return fullPath;
}

void MainWindow::onStartRecording() {
    int cameraId = activeDisplayCamera_;
    if (cameraId < 0) {
        QMessageBox::warning(this, QString::fromUtf8("录像"), QString::fromUtf8("请先打开一个摄像头"));
        return;
    }

    // 检查是否已在录制
    auto it = cameraRecordings_.find(cameraId);
    if (it != cameraRecordings_.end() && it.value()->isRecording) {
        QMessageBox::information(this, QString::fromUtf8("录像"), QString::fromUtf8("该摄像头已在录制中"));
        return;
    }

    // 创建录像目录: 年月日/摄像头ID/
    QString recordDir = getRecordDir(cameraId);
    QDateTime now = QDateTime::currentDateTime();
    QString startTimeStr = now.toString("HHmmss");
    
    // 使用起止时间命名: 开始时间-结束时间.mp4 (结束时重命名)
    QString videoPath = recordDir + "/" + startTimeStr + ".mp4";

    // 从活跃摄像头获取帧尺寸
    auto camIt = cameraWorkers_.find(cameraId);
    int width = 1920, height = 1080;
    if (camIt != cameraWorkers_.end() && camIt->worker) {
        // 尝试从摄像头获取实际尺寸，这里使用默认值
        // 实际尺寸会在第一次写帧时自动调整
    }

    // 创建 VideoWriter
    cv::VideoWriter* writer = new cv::VideoWriter(
        videoPath.toStdString(),
        cv::VideoWriter::fourcc('m','p','4','v'),
        25.0,  // FPS
        cv::Size(width, height)
    );

    if (!writer->isOpened()) {
        QMessageBox::critical(this, QString::fromUtf8("录像"), QString::fromUtf8("无法创建录像文件: ") + videoPath);
        delete writer;
        return;
    }

    // 保存录制信息
    CameraRecording* rec = new CameraRecording();
    rec->cameraId = cameraId;
    rec->videoPath = videoPath;
    rec->writer = writer;
    rec->isRecording = true;
    rec->startTime = now;
    cameraRecordings_[cameraId] = rec;

    ui->startRecordBtn->setEnabled(false);
    ui->stopRecordBtn->setEnabled(true);

    log(QString::fromUtf8("录像"), QString::fromUtf8("开始录制: %1 (摄像头%2)").arg(videoPath).arg(cameraId));
}

void MainWindow::onStopRecording() {
    int cameraId = activeDisplayCamera_;
    auto it = cameraRecordings_.find(cameraId);
    if (it == cameraRecordings_.end() || !it.value()->isRecording) {
        QMessageBox::warning(this, QString::fromUtf8("录像"), QString::fromUtf8("当前没有正在录制的摄像头"));
        return;
    }

    auto rec = it.value();
    // 停止录制
    if (rec->writer && rec->writer->isOpened()) {
        rec->writer->release();
    }
    delete rec->writer;
    rec->writer = nullptr;
    rec->isRecording = false;

    QDateTime now = QDateTime::currentDateTime();
    QString startTimeStr = rec->startTime.toString("HHmmss");
    QString endTimeStr = now.toString("HHmmss");
    QString videoPath = rec->videoPath;

    // 重命名为带起止时间的文件名: 开始时间-结束时间.mp4
    QString newPath = videoPath;
    newPath.replace(".mp4", "-" + endTimeStr + ".mp4");
    
    if (QFile::exists(videoPath)) {
        QFile::rename(videoPath, newPath);
    }

    log(QString::fromUtf8("录像"), QString::fromUtf8("停止录制: %1").arg(newPath));

    cameraRecordings_.erase(it);

    ui->startRecordBtn->setEnabled(true);
    ui->stopRecordBtn->setEnabled(false);
}

void MainWindow::onViewRecordings() {
    QString recordDir = QDir::cleanPath(QDir::currentPath() + "/" + QString::fromStdString(Config::RECORD_DIR));
    QDir dir(recordDir);
    if (!dir.exists()) {
        QMessageBox::information(this, QString::fromUtf8("录像"), QString::fromUtf8("录像目录不存在: ") + recordDir);
        return;
    }
    // 打开文件目录
    QDesktopServices::openUrl(QUrl::fromLocalFile(recordDir));
    log(QString::fromUtf8("录像"), QString::fromUtf8("打开录像目录: %1").arg(recordDir));
}

void MainWindow::onClearOldRecordings() {
    int keepDays = ui->recordDaysSpin->value();
    QDateTime cutoff = QDateTime::currentDateTime().addDays(-keepDays);
    
    QString recordDir = QDir::cleanPath(QDir::currentPath() + "/" + QString::fromStdString(Config::RECORD_DIR));
    QDir dir(recordDir);
    if (!dir.exists()) return;
    
    int deletedCount = 0;
    QFileInfoList dateDirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& dateDirInfo : dateDirs) {
        QDateTime dirTime = QFileInfo(dir, dateDirInfo.fileName()).lastModified();
        if (dirTime < cutoff) {
            QString subDirPath = recordDir + "/" + dateDirInfo.fileName();
            QDir subDir(subDirPath);
            // 删除日期目录下的所有文件
            QFileInfoList files = subDir.entryInfoList(QDir::Files);
            for (const QFileInfo& f : files) {
                QFile::remove(f.absoluteFilePath());
                deletedCount++;
            }
            // 删除空的子目录
            subDir.rmdir(subDirPath);
            // 删除日期目录
            dir.rmdir(dateDirInfo.fileName());
        }
    }
    
    log(QString::fromUtf8("录像"), QString::fromUtf8("已清理 %1 个旧录像文件").arg(deletedCount));
    QMessageBox::information(this, QString::fromUtf8("清理完成"), 
        QString::fromUtf8("已清理 %1 个旧录像文件").arg(deletedCount));
}

// 录像时写入帧
void MainWindow::writeRecordingFrame(int cameraId, const cv::Mat& frame) {
    auto it = cameraRecordings_.find(cameraId);
    if (it == cameraRecordings_.end()) return;
    auto rec = it.value();
    if (!rec || !rec->isRecording) return;
    if (!rec->writer || !rec->writer->isOpened()) return;
    
    // 缩放到1080p
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(1920, 1080));
    rec->writer->write(resized);
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
    
    // 根据不同类别设置颜色
    QString color;
    if (category == QString::fromUtf8("告警") || category == QString::fromUtf8("错误")) {
        color = "#e74c3c";  // 红色 - 告警/错误
    } else if (category == QString::fromUtf8("检测")) {
        color = "#27ae60";  // 绿色 - 检测
    } else if (category == QString::fromUtf8("录像")) {
        color = "#e67e22";  // 橙色 - 录像
    } else if (category == QString::fromUtf8("WebSocket") || category == QString::fromUtf8("MJPEG")) {
        color = "#3498db";  // 蓝色 - 网络相关
    } else if (category == QString::fromUtf8("配置")) {
        color = "#9b59b6";  // 紫色 - 配置
    } else if (category == QString::fromUtf8("系统")) {
        color = "#34495e";  // 深灰 - 系统
    } else {
        color = "#7f8c8d";  // 灰色 - 其他
    }
    
    QString formatted = QString("<span style='color:%1; font-size:22px;'>[%2][%3] %4</span>")
        .arg(color, timestamp, category, message);
    
    ui->logTextEdit->append(formatted);
    // 自动滚动到底部
    //QTextCursor cursor = ui->logTextEdit->textCursor();
    //cursor.movePosition(QTextCursor::End);
    //ui->logTextEdit->setTextCursor(cursor);
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
    //const auto interfaces = QNetworkInterface::allInterfaces();
    //for (const auto& iface : interfaces) {
    //    if (iface.flags() & QNetworkInterface::IsLoopBack) continue;
    //    if (!(iface.flags() & QNetworkInterface::IsUp)) continue;
    //    if (!(iface.flags() & QNetworkInterface::IsRunning)) continue;
    //    for (const auto& addr : iface.addressEntries()) {
    //        if (addr.ip().protocol() == QAbstractSocket::IPv4Protocol &&
    //            addr.ip() != QHostAddress::LocalHost) {
    //            return addr.ip().toString();
    //        }
    //    }
    //}
    // 回退到配置文件中的IP
    return QString::fromStdString(Config::HOST_IP);
}
