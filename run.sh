#!/bin/bash
echo "啟動系統: $(expr substr $(uname -s) 1 10)"

if [ "$1" ]; then
	APP_TYPE=$1
else
	echo "請輸入要啟動的種類"
  exit
fi

if [ ! -d "src/app/${APP_TYPE}" ]; then
  echo "找不到 ${APP_TYPE} 的資料夾"
  exit
fi

if [ "$2" ]; then
  APP_SCRIPT=$2
else
  APP_SCRIPT=${APP_TYPE}
fi

if [ ! -f "src/app/${APP_TYPE}/${APP_SCRIPT}.py" ]; then
  echo "找不到 ${APP_TYPE} 的 ${APP_SCRIPT}.py"
  exit
fi

echo "啟動路徑: src/app/${APP_TYPE}/${APP_SCRIPT}.py"
if [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]; then
  python -m src.app.${APP_TYPE}.${APP_SCRIPT}
else
  python3 -m src.app.${APP_TYPE}.${APP_SCRIPT}
fi
