@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM YOLO11 PPE Detection - Deployment Packaging Script
REM Collects Release build + all dependency DLLs into dist/
REM Target PC needs: NVIDIA GPU + latest driver (CUDA requirement)
REM ============================================================

echo ========================================
echo   Packaging YOLO11 PPE Detection System
echo ========================================

set "ROOT=%~dp0"
set "BUILD_DIR=%ROOT%build"
set "DIST_DIR=%ROOT%dist"
set "QTDIR=C:\Qt\5.15.2\msvc2019_64"
set "OPENCV_DIR=C:\opencv412\opencv\build\x64\vc16\bin"
set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
set "TENSORRT_DIR=C:\tensorRT10\TensorRT-10.1.0.27\lib"

echo.
echo [1/5] Building Release...
cd /d "%ROOT%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM Use CMake if available, otherwise skip build step
where cmake >nul 2>nul
if %ERRORLEVEL% equ 0 (
    if exist "%BUILD_DIR%\CMakeCache.txt" (
        cmake --build . --config Release
    ) else (
        cmake .. -G "Visual Studio 18 2026" -A x64 -DQt5_DIR="%QTDIR%/lib/cmake/Qt5" -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release
    )
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Build failed.
        pause
        exit /b 1
    )
    echo   Build OK
) else (
    echo   CMake not found, assuming yolo11_1.exe already exists
)

REM Create dist directory
echo.
echo [2/5] Creating output directory...
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"
mkdir "%DIST_DIR%\model"
mkdir "%DIST_DIR%\output"

REM Copy exe
echo.
echo [3/5] Copying executable...
if exist "%BUILD_DIR%\Release\yolo11_1.exe" (
    copy /y "%BUILD_DIR%\Release\yolo11_1.exe" "%DIST_DIR%\" >nul
) else if exist "%BUILD_DIR%\bin\Release\yolo11_1.exe" (
    copy /y "%BUILD_DIR%\bin\Release\yolo11_1.exe" "%DIST_DIR%\" >nul
) else (
    echo   [WARNING] yolo11_1.exe not found in build output
    echo   Please build the project first, then copy the exe manually
)
echo   yolo11_1.exe

REM windeployqt
echo.
echo [4/5] Deploying Qt runtime libraries...
if exist "%QTDIR%\bin\windeployqt.exe" (
    "%QTDIR%\bin\windeployqt.exe" "%DIST_DIR%\yolo11_1.exe" --no-angle --no-opengl-sw >nul
    echo   Qt deployed
) else (
    echo   [WARNING] windeployqt.exe not found, install Qt first
)

REM Copy third-party DLLs
echo.
echo [5/5] Copying dependency DLLs...

REM OpenCV
copy /y "%OPENCV_DIR%\opencv_world4120.dll" "%DIST_DIR%\" >nul
if exist "%OPENCV_DIR%\opencv_videoio_ffmpeg4120_64.dll" (
    copy /y "%OPENCV_DIR%\opencv_videoio_ffmpeg4120_64.dll" "%DIST_DIR%\" >nul
)
echo   OpenCV OK

REM CUDA runtime (essential for TensorRT)
copy /y "%CUDA_DIR%\cudart64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\cublas64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\cublasLt64_12.dll" "%DIST_DIR%\" >nul
echo   CUDA Runtime + BLAS OK

REM CUDA NPP (used by TensorRT preprocessing)
copy /y "%CUDA_DIR%\nppc64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppial64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppicc64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppidei64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppif64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppig64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppim64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppist64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppisu64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nppitc64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\npps64_12.dll" "%DIST_DIR%\" >nul
echo   CUDA NPP OK

REM CUDA other deps
copy /y "%CUDA_DIR%\cufft64_11.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\curand64_10.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\cusolver64_11.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\cusparse64_12.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nvrtc64_120_0.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nvrtc-builtins64_124.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nvJitLink_120_0.dll" "%DIST_DIR%\" >nul
copy /y "%CUDA_DIR%\nvfatbin_120_0.dll" "%DIST_DIR%\" >nul
echo   CUDA other deps OK

REM TensorRT
copy /y "%TENSORRT_DIR%\nvinfer_10.dll" "%DIST_DIR%\" >nul
copy /y "%TENSORRT_DIR%\nvinfer_plugin_10.dll" "%DIST_DIR%\" >nul
copy /y "%TENSORRT_DIR%\nvonnxparser_10.dll" "%DIST_DIR%\" >nul
echo   TensorRT OK

REM Copy model file
echo.
copy "%ROOT%model\*.engine" "%DIST_DIR%\model\" 2>nul
if exist "%DIST_DIR%\model\*.engine" (
    echo Model file copied to dist\model\
) else (
    echo [NOTE] No .engine model found in model\
    echo        Place your TensorRT engine file in dist\model\
)

REM Generate README
echo.
(
echo YOLO11 PPE Detection System - Standalone Package
echo ================================================
echo.
echo Requirements:
echo   - Windows 10/11 64-bit
echo   - NVIDIA GPU with latest driver (provides CUDA runtime)
echo.
echo How to use:
echo   1. Place your .engine model file in model\
echo   2. Run yolo11_1.exe
echo.
echo Directory layout:
echo   yolo11_1.exe       - Main executable
echo   model\             - TensorRT engine model (.engine)
echo   output\            - Batch detection results
echo   *.dll              - Runtime dependencies
) > "%DIST_DIR%\README.txt"

echo.
echo ========================================
echo   Packaging complete!
echo   Output: %DIST_DIR%
echo ========================================
pause
