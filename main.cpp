/**
 * @file main.cpp
 * @brief YOLO11 TensorRT 推理系统 - Qt GUI 主入口
 *
 * 本文件启动Qt应用程序, 创建并显示主窗口.
 * 相较于命令行版本, GUI版本提供了更友好的交互方式:
 * - 可视化图像/视频显示
 * - 实时检测结果列表
 * - 置信度和NMS阈值实时调节
 * - 摄像头实时检测
 *
 * 使用方式:
 *   直接运行可执行文件, 在界面中操作
 */

#include <QApplication>
#include <QDir>
#include <QTextCodec>
#include <Windows.h>
#include "mainwindow.hpp"
#include "types.hpp"

int main(int argc, char *argv[]) {
    // Windows UTF-8 控制台支持
    SetConsoleOutputCP(CP_UTF8);
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
    // 注册自定义类型用于跨线程信号槽
    qRegisterMetaType<Detection>("Detection");
    qRegisterMetaType<std::vector<Detection>>("std::vector<Detection>");

    // 创建Qt应用程序(管理事件循环, 处理窗口消息等)
    QApplication app(argc, argv);

    // 将工作目录设为exe所在目录, 确保相对路径(模型文件等)正确解析
    QDir::setCurrent(QApplication::applicationDirPath());

    // 设置应用程序元信息(用于窗口标题等)
    app.setApplicationName("YOLO11 PPE Detection");
    app.setApplicationVersion("1.0");

    // 使用 Fusion 风格, 使运行时界面与 Qt Creator 设计预览一致
    app.setStyle("Fusion");

    // 创建并显示主窗口(支持自由拖拽缩放)
    MainWindow mainWindow;
    mainWindow.show();

    // 进入Qt事件循环(等待用户操作)
    return app.exec();
}
