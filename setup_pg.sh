#!/bin/bash

# 设置 PostgreSQL 安装目录
PREFIX_DIR="/data1/chenxu/projects/pg_install_ml_5438/"

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

cd "./conf"
python pre_warm.py
