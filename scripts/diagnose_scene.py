"""
场景解析定位脚本 —— 排查「雷达有、卫星0、目标0」问题。

只依赖标准库，可直接拷贝到离线机运行：
    python diagnose_scene.py <scene.json> [plan_id]

它不复用 sim 内核（内核会把异常吞掉、只返回半成品），而是逐字段打印
外层列表、analyseParam 里的 id 集合，以及两种口径（精确 == 与 str 归一）
下的 id 交叉匹配结果，直接定位是「字段缺失 / 改名 / id 类型不一致」哪种。
"""
import json
import sys


def cross_match(label, outer_ids, inner_ids):
    """打印外层 id 与 analyseParam id 集合的交叉匹配结果。"""
    exact = [o for o in outer_ids if o in inner_ids]
    str_map = {str(v): v for v in inner_ids}
    normalized = [o for o in outer_ids if str(o) in str_map]
    print(f"  {label}: 外层 id={outer_ids!r} (类型 {[type(o).__name__ for o in outer_ids]})")
    print(f"     analyseParam id 集合={inner_ids!r} (类型 {[type(v).__name__ for v in inner_ids]})")
    print(f"     精确 == 命中 {len(exact)} 个; str 归一命中 {len(normalized)} 个")
    if not exact and normalized:
        print("     >>> 命中数差在这里：id 类型不一致（int vs str）！")
    return exact, normalized


def summarize_id_field(v):
    """把列表/dict 压成 (类型, 长度, id 列表)。"""
    if isinstance(v, dict):
        return "dict", len(v), list(v.keys())
    if isinstance(v, list):
        return "list", len(v), v
    return type(v).__name__, None, v


def main(path, plan_id=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"文件: {path}")
    print(f"顶层 keys: {list(data.keys())}")

    plans = data.get("planInfoList", [])
    print(f"planInfoList 条数 = {len(plans)}")

    for i, p in enumerate(plans):
        if not isinstance(p, dict):
            continue
        pid = p.get("planId") or p.get("associatedTaskId") or p.get("id")
        if plan_id is not None and str(pid) != str(plan_id):
            continue

        print(f"\n===== planInfoList[{i}] (id={pid!r}) =====")

        # 1) 外层实体列表：只看条数和 id
        outer_keys = {
            "radarList": "雷达",
            "satellifeList": "卫星",
            "satelliteList": "卫星(别名)",
            "missileTargetList": "目标",
            "targetList": "目标(别名)",
        }
        for key, name in outer_keys.items():
            if key not in p:
                print(f"  [缺] 外层.{key} ({name})")
                continue
            lst = p[key] if isinstance(p[key], list) else []
            ids = [x.get("id") for x in lst if isinstance(x, dict)]
            print(f"  外层.{key} ({name}): 条数={len(lst)}, id 类型={[type(o).__name__ for o in ids][:5]}")

        # 2) planScene -> analyseParam
        ps = p.get("planScene")
        if not isinstance(ps, str):
            print(f"  [异常] planScene 不是字符串: {type(ps).__name__}")
            continue
        try:
            ap = json.loads(ps).get("analyseParam")
        except Exception as e:
            print(f"  [异常] planScene 解析失败: {e}")
            continue

        if not isinstance(ap, dict):
            print(f"  [异常] analyseParam 不是 dict: {type(ap).__name__} = {ap!r}")
            continue
        print(f"  analyseParam keys: {list(ap.keys())}")
        for key in ("radarInfos", "sateIds", "airTargetIds"):
            if key not in ap:
                print(f"  [缺] analyseParam.{key}  ← 缺失会导致对应实体解析为 0")
            else:
                t, n, _ = summarize_id_field(ap[key])
                print(f"  analyseParam.{key}: 类型={t}, 长度={n}")

        # 3) 雷达交叉匹配（参考：雷达能出实体，说明这个口径是通的）
        radar_list = p.get("radarList", []) or []
        radar_infos = ap.get("radarInfos") or {}
        if isinstance(radar_infos, dict):
            radar_inner = list(radar_infos.keys())
        else:
            radar_inner = radar_infos or []
        cross_match("雷达", [r.get("id") for r in radar_list if isinstance(r, dict)], radar_inner)

        # 4) 卫星交叉匹配
        sate_list = p.get("satellifeList") or p.get("satelliteList") or []
        cross_match("卫星", [s.get("id") for s in sate_list if isinstance(s, dict)], ap.get("sateIds") or [])

        # 5) 目标交叉匹配
        tgt_list = p.get("missileTargetList") or p.get("targetList") or []
        cross_match("目标", [t.get("id") for t in tgt_list if isinstance(t, dict)], ap.get("airTargetIds") or [])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
