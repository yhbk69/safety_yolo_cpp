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

#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;


// ============================================================
// InferenceWorker
// ============================================================

InferenceWorker::InferenceWorker(YoloTrtEngine* engine)
    : engine_(engine)
{
    outputDir_ = QDir::cleanPath(QDir::currentPath() + "/" +
        QString::fromStdString(Config::OUTPUT_DIR));
    QDir().mkpath(outputDir_);
}

void InferenceWorker::processVideo(const QString& path, float confThresh, float nmsThresh) {
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) {
        emit errorOccurred("无法打开视频文件: " + path);
        emit finished();
        return;
    }
    running_ = true;
    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        auto result = processOneFrame(frame, confThresh, nmsThresh);
        emit frameProcessed(result.image, result.detections, 0);
    }
    cap.release();
    emit finished();
}

void InferenceWorker::processCamera(float confThresh, float nmsThresh) {
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        emit errorOccurred("无法打开摄像头, 请检查设备连接");
        emit finished();
        return;
    }
    running_ = true;
    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        auto result = processOneFrame(frame, confThresh, nmsThresh);
        emit frameProcessed(result.image, result.detections, 0);
        if (cv::waitKey(1) == 27) break;
    }
    cap.release();
    emit finished();
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
    engine_->infer(tensor, detections, frame.cols, frame.rows, confThresh, nmsThresh);

    cv::Mat displayImg = frame.clone();
    Postprocessor::drawDetections(displayImg, detections);

    cv::Mat rgb;
    cv::cvtColor(displayImg, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);

    // 环形缓冲区
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        frameBuffer_.push_back(displayImg.clone());
        if (frameBuffer_.size() > Config::RING_BUFFER_FRAMES) {
            frameBuffer_.pop_front();
        }
    }

    // 告警检测
    checkAlert(detections, displayImg.clone());

    // 告警录制后续帧
    if (alertRecording_ && alertRemainingFrames_ > 0) {
        alertBuffer_.push_back(displayImg.clone());
        alertRemainingFrames_--;
        if (alertRemainingFrames_ == 0) {
            // 录制完成，保存文件
            saveAlertFiles(QUuid::createUuid().toString(QUuid::WithoutBraces),
                          pendingAlarmType_);
            alertRecording_ = false;
            alertBuffer_.clear();
            pendingAlarmType_.clear();
        }
    }

    FrameResult result;
    result.image = qimg.copy();
    result.detections = std::move(detections);
    return result;
}

void InferenceWorker::checkAlert(
    const std::vector<Detection>& detections, const cv::Mat& annotatedFrame)
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
            alertBuffer_ = frameBuffer_;
        }
        alertBuffer_.push_back(annotatedFrame);

        return;
    }
}

void InferenceWorker::saveAlertFiles(const QString& alarmId, const QString& alarmType) {
    if (alertBuffer_.empty()) return;

    int frameW = alertBuffer_.back().cols;
    int frameH = alertBuffer_.back().rows;
    if (frameW <= 0 || frameH <= 0) return;

    // 文件名: alarm_<alarmId>_<type>
    QString baseName = QString("alarm_%1_%2").arg(alarmId).arg(alarmType);
    QString videoPath = outputDir_ + "/" + baseName + ".mp4";
    QString imagePath = outputDir_ + "/" + baseName + ".jpg";

    // 保存视频(MP4)
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(videoPath.toStdString(), fourcc, 30.0,
                           cv::Size(frameW, frameH));
    if (writer.isOpened()) {
        for (auto& f : alertBuffer_) {
            writer.write(f);
        }
        writer.release();
    }

    // 保存截图(第一帧)
    cv::imwrite(imagePath.toStdString(), alertBuffer_.front());

    // 构造告警 JSON
    QJsonObject data;
    data["alarm_id"]   = alarmId;
    data["alarm_type"] = alarmType;
    data["timestamp"]  = QDateTime::currentDateTime().toMSecsSinceEpoch();
    data["video_url"]  = QString("http://%1:%2/%3").arg(
        QString::fromStdString(Config::HOST_IP)).arg(Config::HTTP_PORT).arg(baseName + ".mp4");
    data["image_url"]  = QString("http://%1:%2/%3").arg(
        QString::fromStdString(Config::HOST_IP)).arg(Config::HTTP_PORT).arg(baseName + ".jpg");

    QJsonObject root;
    root["type"] = "alarm";
    root["data"] = data;

    QString alertJson = QString::fromUtf8(
        QJsonDocument(root).toJson(QJsonDocument::Compact));

    emit alertSaved(videoPath, imagePath, alertJson);
}


// ============================================================
// MainWindow
// ============================================================

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , engine_(nullptr)
    , workerThread_(nullptr)
    , worker_(nullptr)
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

    // 确保输出目录存在
    QDir().mkpath(QString::fromStdString(Config::OUTPUT_DIR));

    startWebSocketServer();
    if (fs::exists(Config::MODEL_PATH)) onLoadModel();

    // 设置批量推理复选框初始状态
    ui->batchInferenceCheck->setChecked(Config::USE_BATCH_INFERENCE);

    // 初始日志
    log("系统", "YOLO11 PPE 检测系统已启动");
    log("配置", QString("模型路径: %1").arg(QString::fromStdString(Config::MODEL_PATH)));
}

MainWindow::~MainWindow() {
    if (isProcessing_) safeStopWorker();
    // 清理所有待确认的告警定时器
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
    delete ui;
}

void MainWindow::setupConnections() {
    connect(ui->openImageBtn,  &QPushButton::clicked, this, &MainWindow::onOpenImage);
    connect(ui->openVideoBtn,  &QPushButton::clicked, this, &MainWindow::onOpenVideo);
    connect(ui->cameraBtn,     &QPushButton::clicked, this, &MainWindow::onOpenCamera);
    connect(ui->folderBtn,     &QPushButton::clicked, this, &MainWindow::onOpenFolder);
    connect(ui->stopBtn,       &QPushButton::clicked, this, &MainWindow::onStopProcessing);
    connect(ui->browseModelBtn, &QPushButton::clicked, this, &MainWindow::onBrowseModel);
    connect(ui->loadModelBtn,   &QPushButton::clicked, this, &MainWindow::onLoadModel);
    connect(ui->confSlider, &QSlider::valueChanged, this, &MainWindow::onConfThresholdChanged);
    connect(ui->nmsSlider,  &QSlider::valueChanged, this, &MainWindow::onNmsThresholdChanged);
    connect(ui->batchInferenceCheck, &QCheckBox::toggled, this, &MainWindow::onBatchInferenceToggled);
    connect(ui->clearLogBtn, &QPushButton::clicked, this, [this]() {
        ui->logTextEdit->clear();
        log("系统", "日志已清空");
    });
    connect(ui->actionOpenImage, &QAction::triggered, this, &MainWindow::onOpenImage);
    connect(ui->actionOpenVideo, &QAction::triggered, this, &MainWindow::onOpenVideo);
    connect(ui->actionOpenCamera, &QAction::triggered, this, &MainWindow::onOpenCamera);
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
            QString request = QString::fromUtf8(socket->readAll());
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

    // 重发给所有连接的客户端
    for (auto* client : wsClients_) {
        client->sendTextMessage(it->jsonMessage);
    }
    statusMessageLabel_->setText(
        QString("重发告警 %1").arg(alarmId.left(8)));
    log("告警", QString("重发告警: %1").arg(alarmId.left(8)));
}

// ============================================================
// 告警处理
// ============================================================
void MainWindow::onAlertSaved(const QString& videoPath, const QString& imagePath,
                               const QString& alertJson) {
    // 解析 alarm_id
    QJsonDocument doc = QJsonDocument::fromJson(alertJson.toUtf8());
    QString alarmId = doc.object()["data"].toObject()["alarm_id"].toString();
    if (alarmId.isEmpty()) return;

    // 广播给所有 WebSocket 客户端
    for (auto* client : wsClients_) {
        client->sendTextMessage(alertJson);
    }

    // 启动 ACK 等待定时器
    auto* timer = new QTimer(this);
    timer->setSingleShot(false);
    connect(timer, &QTimer::timeout, this, [this, alarmId]() {
        retryAlarm(alarmId);
    });

    PendingAlarm pending;
    pending.jsonMessage = alertJson;
    pending.retryTimer = timer;
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

// ============================================================
// 推理模式
// ============================================================
void MainWindow::onOpenImage() {
    if (!engine_) { QMessageBox::warning(this, "提示", "请先加载模型"); return; }
    if (isProcessing_) onStopProcessing();
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
    if (!engine_) { QMessageBox::warning(this, "提示", "请先加载模型"); return; }
    if (isProcessing_) onStopProcessing();

    QString filePath = QFileDialog::getOpenFileName(
        this, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)");
    if (filePath.isEmpty()) return;

    log("检测", QString("打开视频: %1").arg(filePath));
    statusMessageLabel_->setText("正在处理视频...");
    enableControls(false);
    ui->stopBtn->setEnabled(true);
    isProcessing_ = true;

    workerThread_ = new QThread(this);
    worker_ = new InferenceWorker(engine_.get());
    worker_->moveToThread(workerThread_);

    connect(worker_, &InferenceWorker::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(worker_, &InferenceWorker::alertSaved, this, &MainWindow::onAlertSaved);
    connect(worker_, &InferenceWorker::finished, this, &MainWindow::onWorkerFinished);
    connect(worker_, &InferenceWorker::errorOccurred, this, &MainWindow::onWorkerError);
    connect(workerThread_, &QThread::started, this, [this, filePath]() {
        worker_->processVideo(filePath, confThreshold_, nmsThreshold_);
    });
    connect(workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
    workerThread_->start();
}

void MainWindow::onOpenCamera() {
    if (!engine_) { QMessageBox::warning(this, "提示", "请先加载模型"); return; }
    if (isProcessing_) onStopProcessing();

    log("检测", "启动摄像头推理");
    statusMessageLabel_->setText("正在启动摄像头...");
    enableControls(false);
    ui->stopBtn->setEnabled(true);
    isProcessing_ = true;

    workerThread_ = new QThread(this);
    worker_ = new InferenceWorker(engine_.get());
    worker_->moveToThread(workerThread_);

    connect(worker_, &InferenceWorker::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(worker_, &InferenceWorker::alertSaved, this, &MainWindow::onAlertSaved);
    connect(worker_, &InferenceWorker::finished, this, &MainWindow::onWorkerFinished);
    connect(worker_, &InferenceWorker::errorOccurred, this, &MainWindow::onWorkerError);
    connect(workerThread_, &QThread::started, this, [this]() {
        worker_->processCamera(confThreshold_, nmsThreshold_);
    });
    connect(workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
    workerThread_->start();
}

void MainWindow::onOpenFolder() {
    if (!engine_) { QMessageBox::warning(this, "提示", "请先加载模型"); return; }
    if (isProcessing_) onStopProcessing();

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

void MainWindow::onStopProcessing() { safeStopWorker(); }

void MainWindow::safeStopWorker() {
    if (!isProcessing_ || !workerThread_ || !worker_) return;
    worker_->stop();
    workerThread_->quit();
    if (!workerThread_->wait(3000)) {
        workerThread_->terminate();
        workerThread_->wait();
    }
    worker_ = nullptr;
    workerThread_ = nullptr;
    isProcessing_ = false;
    enableControls(true);
    ui->stopBtn->setEnabled(false);
}

void MainWindow::onFrameProcessed(QImage image, std::vector<Detection> detections, double elapsedMs) {
    updateDisplay(image);
    updateDetectionList(detections, elapsedMs);
    if (elapsedMs > 0) {
        double fps = 1000.0 / elapsedMs;
        fpsLabel_->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
        // 每10帧输出一次检测日志
        static int frameCount = 0;
        if (++frameCount % 10 == 0) {
            log("检测", QString("检测到 %1 个目标, 耗时: %2ms, FPS: %3")
                .arg(detections.size()).arg(elapsedMs, 0, 'f', 1).arg(fps, 0, 'f', 1));
        }
    }
}

void MainWindow::onWorkerFinished() {
    if (workerThread_) {
        workerThread_->quit();
        workerThread_->wait();
        worker_ = nullptr;
        workerThread_ = nullptr;
    }
    isProcessing_ = false;
    enableControls(true);
    ui->stopBtn->setEnabled(false);
    statusMessageLabel_->setText("处理完成");
    fpsLabel_->setText("FPS: --");
    log("系统", "处理任务已完成");
}

void MainWindow::onWorkerError(const QString& message) {
    log("错误", message);
    QMessageBox::critical(this, "处理错误", message);
    onWorkerFinished();
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
    pixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->displayLabel->setPixmap(pixmap);
}

void MainWindow::updateDetectionList(const std::vector<Detection>& detections, double elapsedMs) {
    ui->resultListWidget->clear();
    if (detections.empty()) {
        ui->resultListWidget->addItem("未检测到目标");
        ui->totalCountLabel->setText("目标总数: 0");
        timeLabel_->setText(QString("耗时: %1ms").arg(elapsedMs, 0, 'f', 1));
        return;
    }

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
    if (isProcessing_) safeStopWorker();
    event->accept();
}

// ============================================================
// 批量推理开关
// ============================================================
void MainWindow::onBatchInferenceToggled(bool checked) {
    // 注意: 批量推理配置在编译时固定, 此处仅提示用户
    if (checked) {
        statusMessageLabel_->setText("批量推理已启用 (batch=4, 需重新编译生效)");
        log("配置", "批量推理已启用 (batch=4)");
    } else {
        statusMessageLabel_->setText("批量推理已禁用 (需重新编译生效)");
        log("配置", "批量推理已禁用");
    }
    qDebug() << "Batch inference toggled:" << checked;
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
