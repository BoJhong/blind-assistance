# blind-assistance

## 倉庫介紹

這是一個用於視障者的盲人輔助系統，透過D435i深度攝影機進行深度感測，並在Jetson Nano上進行影像處理，最後透過有源蜂鳴器進行障礙物、坑洞提示。

## 運行方式

1. 安裝相關套件
   ```pip3 install -r requirements.txt```
2. 在./src/app找好要運行的`資料夾名稱`和`主程式檔案名稱`。並把該資料夾裡面的`config.example.toml`
   複製一份並命名為`config.toml`，並根據自己的需求修改裡面的參數。
3. 假設我要運行的程式在`product`資料夾底下的`product.py`
   ，則在`該專案最頂層目錄`執行以下指令：

* Windows: ```python -m src.app.product.product```
* Linux: ```python3 -m src.app.product.product```