"""平台接口连通性探针 —— 离线机器上对真实平台做接口级联调。

不依赖模型权重、不依赖 sim/torch，只用 requests + 标准库，可以在只装了基础
依赖的离线机器上单独跑。推送的是**假数据**，用于验证：接口路径对不对、方法
对不对、鉴权要不要、平台收不收。

用法::

    # 全部接口探一遍（默认读 use_cases/config/algo.yaml 里的 PLATFORM_BASE_URL）
    python test/probe_platform.py

    # 指定平台地址
    python test/probe_platform.py --base-url http://192.168.1.20:8080

    # 只探基线接口
    python test/probe_platform.py --only baseline

    # 离线机器上如果配了代理环境变量，务必绕开它
    python test/probe_platform.py --no-proxy

    # 只打印将要发送的报文，不真发（避免往真实平台写脏数据）
    python test/probe_platform.py --dry-run

⚠️ 这些请求会真的落到平台上。taskId 统一带 ``PROBE_`` 前缀且含时间戳，方便
事后在平台侧识别与清理。
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_PROJ_ROOT = Path(__file__).resolve().parent.parent

try:
    import requests
except ImportError:
    requests = None


# ============================================================
# 探测目标 —— 与 use_cases/pusher.py 逐字段对齐
# ============================================================

def build_endpoints(task_id, files):
    """返回 [(名称, 方法, 路径, 报文)]，报文结构与 Pusher 实际发送的一致。"""
    return [
        (
            "metrics",
            "POST", "/prod-api/agent/train/record",
            {"taskId": task_id, "episode": 1, "reward": 1.5, "loss": 0.42},
        ),
        (
            "weights",
            "POST", "/prod-api/agent/model/record",
            {
                "taskId": task_id, "planId": 867, "episode": 10,
                "absModelPath": files["model_dir"], "modelType": 1,
            },
        ),
        (
            "eval",
            "POST", "/api/receive/eval_result",
            {
                "taskId": task_id,
                "timeSeriesFile": files["eval_records"],
                "evalFile": files["eval_metric"],
                "coverage": 63.42,
            },
        ),
        (
            "baseline",
            "POST", "/api/receive/compare_eval_result",
            {
                "taskId": task_id, "taskName": "baseline",
                "timeSeriesFile": files["baseline_records"],
                "evalFile": files["baseline_metric"],
                "coverage": 51.08,
            },
        ),
        (
            "status",
            "POST", "/prod-api/agent/train/status",
            {
                "task_id": task_id, "status": "failed",
                "error_msg": "connectivity probe, please ignore",
            },
        ),
    ]


# ============================================================
# 环境体检
# ============================================================

PROXY_VARS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
]


def check_env():
    """打印可能影响离线连通的环境变量。"""
    print("-" * 66)
    print("[环境体检]")
    found = {k: os.environ[k] for k in PROXY_VARS if os.environ.get(k)}
    if found:
        print("  检测到代理环境变量（离线机器上这是最常见的假性『网络不通』）:")
        for k, v in found.items():
            print(f"    {k} = {v}")
        print("  → 若探测失败，请加 --no-proxy 重试，或把平台 IP 加进 NO_PROXY")
    else:
        print("  未检测到代理环境变量  [OK]")
    print(f"  requests 可用: {requests is not None}")
    if requests is not None:
        print(f"  requests 版本: {requests.__version__}")


def check_tcp(base_url, timeout=5):
    """先做 TCP 层可达性检查，把『网络不通』与『接口不对』分开。"""
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print("-" * 66)
    print(f"[TCP 层] 尝试连接 {host}:{port}")
    started = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            cost = (time.time() - started) * 1000
            print(f"  TCP 握手成功，耗时 {cost:.1f} ms  [OK]")
            return True
    except OSError as e:
        print(f"  TCP 握手失败: {e}  [FAIL]")
        print("  → 这是网络层问题，与 GET/POST 无关。依次检查：")
        print("    1) 平台服务是否已启动、监听端口是否正确")
        print("    2) 平台是否只监听 127.0.0.1（跨机访问需监听 0.0.0.0）")
        print("    3) 本机/目标机防火墙是否放行该端口")
        return False


# ============================================================
# 探测
# ============================================================

def make_session(no_proxy):
    session = requests.Session()
    if no_proxy:
        # trust_env=False 让 requests 忽略 HTTP_PROXY/HTTPS_PROXY 等环境变量
        session.trust_env = False
        session.proxies = {}
    return session


def diagnose(status_code):
    """把状态码翻译成可操作的结论。"""
    if status_code == 200:
        return "[OK]   平台已接收"
    if status_code in (201, 202, 204):
        return "[OK]   平台已接收（非 200 成功码）"
    if status_code == 404:
        return ("[FAIL] 路径不存在 —— 检查是否缺少网关前缀（注意 eval/baseline "
                "两个接口没有 prod-api 前缀，与其余接口不一致）")
    if status_code == 405:
        return "[FAIL] 方法不允许 —— 这才是真正的 GET/POST 问题，问后端要正确方法"
    if status_code in (401, 403):
        return "[FAIL] 鉴权被拒 —— 平台需要 token/cookie，Pusher 目前不带任何认证头"
    if status_code == 415:
        return "[FAIL] 媒体类型不支持 —— 平台可能要 form 而非 json"
    if status_code == 422:
        return "[WARN] 路径通了但报文字段不合平台校验 —— 拿响应体对字段名"
    if 500 <= status_code < 600:
        return "[WARN] 路径通了，平台内部报错 —— 连通性没问题，看平台日志"
    return "[WARN] 未预期状态码"


def probe(session, base_url, name, method, path, payload, timeout, dry_run):
    url = f"{base_url.rstrip('/')}{path}"
    print("-" * 66)
    print(f"[{name}] {method} {url}")
    print(f"  报文: {json.dumps(payload, ensure_ascii=False)}")

    if dry_run:
        print("  [DRY-RUN] 未实际发送")
        return None

    try:
        resp = session.request(method, url, json=payload, timeout=timeout)
    except Exception as e:
        print(f"  请求异常: {type(e).__name__}: {e}  [FAIL]")
        hint = _exception_hint(e)
        if hint:
            print(f"  → {hint}")
        return None

    body = (resp.text or "").strip()
    if len(body) > 300:
        body = body[:300] + " ...(截断)"
    print(f"  HTTP {resp.status_code}  {diagnose(resp.status_code)}")
    print(f"  响应: {body}")
    return resp.status_code


def _exception_hint(e):
    name = type(e).__name__
    text = str(e).lower()
    if "proxy" in name.lower() or "proxy" in text:
        return "代理导致，加 --no-proxy 重试"
    if name in ("ConnectTimeout", "ConnectionError"):
        return ("连不上。TCP 层若已通过，多半是代理拦截（--no-proxy）或 "
                "URL scheme/端口写错")
    if name == "ReadTimeout":
        return "连上了但平台没在超时内回包，可用 --timeout 调大"
    if "ssl" in text or "certificate" in text:
        return "HTTPS 证书问题。离线自签场景可临时用 http，或让后端换 http"
    return None


# ============================================================
# 入口
# ============================================================

def default_base_url():
    """从 algo.yaml 读默认平台地址，读不到就回退到本机 8080。"""
    cfg = _PROJ_ROOT / "use_cases" / "config" / "algo.yaml"
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("PLATFORM_BASE_URL:"):
                    return stripped.split(":", 1)[1].strip().strip('"\'')
    except OSError:
        pass
    return "http://127.0.0.1:8080"


def resolve_files(records_dir):
    """给报文里的文件路径字段填值：有真产物就用真的，没有就用占位路径。"""
    base = Path(records_dir)
    names = {
        "eval_records": "PROBE_eval_records.json",
        "eval_metric": "PROBE_eval_metric.json",
        "baseline_records": "PROBE_baseline_records.json",
        "baseline_metric": "PROBE_baseline_metric.json",
    }
    files = {k: str((base / v).resolve()) for k, v in names.items()}
    files["model_dir"] = str((_PROJ_ROOT / "models" / "PROBE").resolve())

    # 若目录下已有真实产物（例如刚跑过 test_baseline_push.py），优先用真路径
    for key, suffix in (
        ("eval_records", "_eval_records.json"),
        ("eval_metric", "_eval_metric.json"),
        ("baseline_records", "_baseline_records.json"),
        ("baseline_metric", "_baseline_metric.json"),
    ):
        hits = sorted(base.glob(f"*{suffix}")) if base.exists() else []
        if hits:
            files[key] = str(hits[-1].resolve())
    return files


def main():
    parser = argparse.ArgumentParser(description="平台接口连通性探针")
    parser.add_argument("--base-url", default=default_base_url(),
                        help="平台地址，默认取 algo.yaml 的 PLATFORM_BASE_URL")
    parser.add_argument("--only", default="",
                        help="只探某个接口: metrics/weights/eval/baseline/status")
    parser.add_argument("--no-proxy", action="store_true",
                        help="忽略 HTTP_PROXY 等环境变量（离线机器建议开）")
    parser.add_argument("--timeout", type=float, default=8.0, help="超时秒数")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印报文，不实际发送")
    parser.add_argument("--records-dir",
                        default=str(_PROJ_ROOT / "eval_records"),
                        help="产物目录，用于填充报文里的文件路径字段")
    args = parser.parse_args()

    task_id = f"PROBE_{time.strftime('%Y%m%d_%H%M%S')}"

    print("=" * 66)
    print("平台接口连通性探针")
    print("=" * 66)
    print(f"  平台地址: {args.base_url}")
    print(f"  探针 taskId: {task_id}  （平台侧可据此识别/清理测试数据）")

    check_env()

    if requests is None and not args.dry_run:
        print("\n[FAIL] 未安装 requests，无法发送。pip install requests")
        return 2

    tcp_ok = check_tcp(args.base_url, timeout=args.timeout)
    if not tcp_ok and not args.dry_run:
        print("\n[结论] TCP 层就不通，先解决网络可达性，无需再看 HTTP 层。")
        return 1

    files = resolve_files(args.records_dir)
    endpoints = build_endpoints(task_id, files)
    if args.only:
        endpoints = [e for e in endpoints if e[0] == args.only]
        if not endpoints:
            print(f"\n[FAIL] 未知接口名: {args.only}")
            return 2

    session = make_session(args.no_proxy) if requests else None
    if args.no_proxy:
        print("  已启用 --no-proxy（忽略代理环境变量）")

    results = {}
    for name, method, path, payload in endpoints:
        results[name] = probe(session, args.base_url, name, method, path,
                              payload, args.timeout, args.dry_run)

    if args.dry_run:
        print("\n[DRY-RUN] 未发送任何请求")
        return 0

    print("=" * 66)
    print("[汇总]")
    ok = 0
    for name, code in results.items():
        label = f"HTTP {code}" if code is not None else "无响应"
        mark = "OK  " if code and 200 <= code < 300 else "FAIL"
        ok += 1 if code and 200 <= code < 300 else 0
        print(f"  [{mark}] {name:<9} {label}")
    print(f"  通过 {ok}/{len(results)}")
    print("=" * 66)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
