#!/usr/bin/env bash
# =============================================================================
# API 接口 curl 自测试 —— 覆盖 API/routers 下全部 8 个接口
#
# 前置条件：
#   1. 服务已启动（uvicorn main:app，默认 0.0.0.0:8000）
#   2. scene.json 存在（脚本默认取仓库根目录 scene.json，plan_id=867）
#
# 重要说明：
#   - scene_url 是「平台侧已下载到本机的临时场景文件路径」，服务在任务结束后
#     会删除该文件（见 TrainHandler._cleanup_scene）。因此脚本先把 scene.json
#     复制到临时目录再下发，避免删除仓库原件。
#   - train/rl、train/il 为异步接口：立即返回 200「starting」，真实训练在后台执行。
#   - eval/start 需要 load_dir 指向真实训练好的权重，否则返回 500（属预期）。
#   - baseline_eval/start 无权重依赖，可安全自测，返回基线覆盖率等字段。
#
# 用法：
#   bash scripts/curl_selftest.sh            # 全部 8 个接口
#   bash scripts/curl_selftest.sh control    # 只跑 4 个控制接口（最安全）
# =============================================================================

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PREFIX="/api/v1"
# 仓库根目录（用 Windows 正斜杠路径，Python 的 open() 可直接识别；Git Bash 的 cp 亦兼容）
PROJECT_ROOT="${PROJECT_ROOT:-C:/webace_2026/14s/code/webace-3}"
SCENE_SRC="$PROJECT_ROOT/scene.json"     # plan_id=867
LOAD_DIR="${LOAD_DIR:-$PROJECT_ROOT/load_dir}"
TMP_DIR="$PROJECT_ROOT/temp_scenes"
mkdir -p "$TMP_DIR"

# 复制一份场景文件给本次请求（避免服务端任务结束后删除仓库原件），返回 Windows 正斜杠路径
mk_scene() {
    local tid="$1"
    local dst="$TMP_DIR/${tid}.json"
    cp "$SCENE_SRC" "$dst"
    echo "$dst"
}

req() {
    local label="$1"; shift
    echo
    echo "====================================================================="
    echo "▶ $label"
    echo "====================================================================="
    curl -sS -X POST "$BASE_URL$PREFIX$1" \
        -H "Content-Type: application/json" \
        -d "$2"
    echo
}

SCOPE="${1:-all}"

# ---- 4 个控制接口（纯标志文件，最安全） ----
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "control" ]; then
    TID="curl_ctrl_001"
    req "POST /train/pause"      /train/pause      "{\"task_id\":\"$TID\"}"
    req "POST /train/speed"      /train/speed      "{\"task_id\":\"$TID\",\"speed\":2.0}"
    req "POST /train/resume"     /train/resume     "{\"task_id\":\"$TID\"}"
    req "POST /train/terminate"  /train/terminate  "{\"task_id\":\"$TID\"}"
fi

if [ "$SCOPE" != "all" ]; then exit 0; fi

# ---- 基准推演（无权重依赖，安全） ----
BASE_TID="curl_base_001"
BASE_SCENE="$(mk_scene "$BASE_TID")"
req "POST /baseline_eval/start" /baseline_eval/start \
    "{\"task_id\":\"$BASE_TID\",\"plan_id\":867,\"scene_url\":\"$BASE_SCENE\",\"algorithm\":\"qmix\"}"

# ---- RL 训练（异步，后台真实训练；返回 starting） ----
RL_TID="curl_rl_001"
RL_SCENE="$(mk_scene "$RL_TID")"
req "POST /train/rl" /train/rl \
    "{\"task_id\":\"$RL_TID\",\"plan_id\":867,\"load_dir\":\"$LOAD_DIR\",\"scene_url\":\"$RL_SCENE\",\"max_episodes\":3,\"algorithm\":\"qmix\",\"rl_num_workers\":0,\"hyperparameters\":{}}"

# ---- IL 训练（异步，后台真实训练；返回 starting） ----
IL_TID="curl_il_001"
IL_SCENE="$(mk_scene "$IL_TID")"
req "POST /train/il" /train/il \
    "{\"task_id\":\"$IL_TID\",\"plan_id\":867,\"load_dir\":\"$LOAD_DIR\",\"scene_url\":\"$IL_SCENE\",\"algorithm\":\"drqn\",\"epochs\":2,\"il_num_workers\":0,\"hyperparameters\":{}}"

# ---- 推演评估（同步；load_dir 需指向真实权重，否则 500 属预期） ----
EVAL_TID="curl_eval_001"
EVAL_SCENE="$(mk_scene "$EVAL_TID")"
req "POST /eval/start" /eval/start \
    "{\"task_id\":\"$EVAL_TID\",\"plan_id\":867,\"scene_url\":\"$EVAL_SCENE\",\"algorithm\":\"qmix\",\"load_dir\":\"$LOAD_DIR\",\"max_episodes\":1}"
