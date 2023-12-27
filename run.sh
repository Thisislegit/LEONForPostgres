#!/bin/bash

# Check if ./log/ exists, if not create it
if [ ! -d "./log/" ]; then
    mkdir "./log/"
fi

# Check if ./logs/ exists, if not create it
if [ ! -d "./logs/" ]; then
    mkdir "./logs/"
fi

# 进入 log 目录并删除 model.pth 和 message.pkl
cd log
rm model.pth
rm messages.pkl
cd ..

# 启动 leon_server2.py
python leon_server_v3.py
