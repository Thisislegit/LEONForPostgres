#!/bin/bash

# Check if ./log/ exists, if not create it
if [ ! -d "./log/" ]; then
    mkdir "./log/"
fi

# Check if ./logs/ exists, if not create it
if [ ! -d "./logs/" ]; then
    mkdir "./logs/"
fi

# PREFIX_DIR="/data1/wyz/DATA_1120"

# cd "$PREFIX_DIR"

# 重启 PostgreSQL
# ./bin/pg_ctl -D ./data restart
# ./bin/pg_ctl -D ./data1 restart
# ./bin/pg_ctl -D ./data2 restart
# ./bin/pg_ctl -D ./data3 restart

# cd "/data1/wyz/online/LEONForPostgres/conf"
# python pre_warm.py

# cd ..


# 进入 log 目录并删除 model.pth 和 message.pkl
cd log
rm model.pth
rm messages.pkl
cd ..

# 启动 leon_server2.py
python leon_server_v4.py
