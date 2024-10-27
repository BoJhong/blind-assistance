@echo off

rem 檢查參數1是否存在
if "%1"=="" (
    echo 請輸入要啟動的種類
    exit /b
)

set APP_TYPE=%1

rem 檢查資料夾是否存在
if not exist "src\app\%APP_TYPE%" (
    echo 找不到 %APP_TYPE% 的資料夾
    exit /b
)

rem 檢查第二個參數是否存在
if "%2"=="" (
    set APP_SCRIPT=%APP_TYPE%
) else (
    set APP_SCRIPT=%2%
)

rem 檢查 Python 腳本文件是否存在
if not exist "src\app\%APP_TYPE%\%APP_SCRIPT%.py" (
    echo 找不到 %APP_TYPE% 的 %APP_SCRIPT%.py
    exit /b
)

echo 啟動路徑: src\app\%APP_TYPE%\%APP_SCRIPT%.py

python -m src.app.%APP_TYPE%.%APP_SCRIPT%
