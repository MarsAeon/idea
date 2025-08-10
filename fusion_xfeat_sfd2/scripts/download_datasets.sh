#!/usr/bin/env bash
set -e
# 示例脚本：实际需根据许可与数据源补全
DATA_ROOT=${1:-"datasets"}
mkdir -p ${DATA_ROOT}

echo "[Info] 请根据实际数据源填写下载命令 (Aachen / RobotCar / HPatches / MegaDepth)"
echo "[Info] Aachen: 需要注册访问; RobotCar: 官方站点; HPatches: 直接wget; MegaDepth: 参照官方链接"

# HPatches 示例
# wget -c http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -O ${DATA_ROOT}/hpatches.tar.gz
# tar -xf ${DATA_ROOT}/hpatches.tar.gz -C ${DATA_ROOT}

# 生成完成标记
# touch ${DATA_ROOT}/.download_complete
