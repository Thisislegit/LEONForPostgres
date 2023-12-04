#!/bin/bash

# 进入 log 目录并删除 model.pth 和 message.pkl
cd log
rm model.pth
rm messages.pkl
cd ..

# 启动 leon_server2.py
python leon_server2.py
