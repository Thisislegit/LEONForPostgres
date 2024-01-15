#!/bin/bash

# 设置 PostgreSQL 安装目录
PREFIX_DIR="/data1/wyz/DATA_1120"

# 执行 configure
./configure --prefix="$PREFIX_DIR"

# 清理之前的构建
make clean

# 使用多线程进行构建
make -j

# 安装
make install

# 进入目录
cd "$PREFIX_DIR"

# 重启 PostgreSQL
./bin/pg_ctl -D ./data restart
./bin/pg_ctl -D ./data1 restart
./bin/pg_ctl -D ./data2 restart
./bin/pg_ctl -D ./data3 restart
./bin/pg_ctl -D ./data4 restart
./bin/pg_ctl -D ./data5 restart

cd "/data1/wyz/online/LEONForPostgres/conf"
python pre_warm.py
