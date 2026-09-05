"""打印一帧 ZMQ 态势推送的真实内容，并对比「旧协议（无 episode/step）」vs「新协议（带 episode/step）」。

用法（仓库根目录）::

    .venv/Scripts/python.exe scripts/demo_zmq_situation.py
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_PB2 = os.path.join(_ROOT, "services", "zmq", "proto", "ProtoStruct_pb2.py.bak")
NEW_PB2 = os.path.join(_ROOT, "services", "zmq", "proto", "ProtoStruct_pb2.py")
_PY = sys.executable

# 与 SituationPublisher.push_frame 完全一致的组装逻辑，只是把「发送」换成「打印 + 序列化」。
_DUMP_SNIPPET = r'''
import sys
from importlib.machinery import SourceFileLoader
from types import SimpleNamespace

mod = SourceFileLoader('pb2mod', sys.argv[1]).load_module()
with_new = '--with-new-fields' in sys.argv


def _id(s):
    digits = ''.join(ch for ch in str(s) if ch.isdigit())
    return int(digits) if digits else 0


def build():
    # ---- 模拟 raw_obs（AgentObservation 精简版） ----
    def equip(eid, typ, lon, lat, alt):
        return SimpleNamespace(str_equip_id=eid, type=typ,
                               longitude=lon, latitude=lat, altitude=alt)
    def track(tid, lon, lat, alt, typ):
        return SimpleNamespace(str_system_track_no=tid, longitude=lon,
                               latitude=lat, altitude=alt, type=typ)
    def cmd(eid, tid):
        return SimpleNamespace(str_equip_id=eid, str_target_id=tid)

    raw_obs = SimpleNamespace(
        dict_equip_state={
            'R01': equip('R01', 1, 120.15, 30.28, 500.0),
            'R02': equip('R02', 1, 121.05, 31.12, 480.0),
            'W01': equip('W01', 2, 119.98, 30.05, 8000.0),
        },
        dict_system_track={
            'T01': track('T01', 122.5, 28.5, 15000.0, 6),
            'T02': track('T02', 123.0, 27.8, 12000.0, 1),
            'T03': track('T03', 121.8, 29.9, 18000.0, 2),
        },
    )
    # 有效动作（str_target_id 非空才进 Detection），混一个空目标演示过滤
    valid_cmds = [cmd('R01', 'T01'), cmd('R01', 'T02'), cmd('R02', 'T03'),
                  cmd('W01', '')]   # 空目标 → 被过滤
    return raw_obs, valid_cmds


raw_obs, valid_cmds = build()
msg = mod.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
msg.CurrentTime = 12.5
msg.DataType = 0
if with_new:
    msg.EpisodeIdx = 7
    msg.StepIdx = 33
msg.PubCtrl.MsgHeader.MsgType = 12290

for eid, es in raw_obs.dict_equip_state.items():
    e = msg.EquipPos.add()
    e.EquipID = _id(es.str_equip_id); e.EquipType = es.type; e.Time = msg.CurrentTime
    e.GeoPos.X = float(es.longitude); e.GeoPos.Y = float(es.latitude); e.GeoPos.Z = float(es.altitude)
for tid, tb in raw_obs.dict_system_track.items():
    t = msg.TargetPos.add()
    t.TargetID = _id(tb.str_system_track_no); t.Time = msg.CurrentTime
    t.GeoPos.X = float(tb.longitude); t.GeoPos.Y = float(tb.latitude); t.GeoPos.Z = float(tb.altitude)
for c in valid_cmds:
    if c.str_target_id and c.str_target_id not in ('', '0'):
        d = msg.Detection.add()
        d.EquipID = _id(c.str_equip_id); d.TargetID = _id(c.str_target_id)

print('=' * 70)
print(('新协议（带 EpisodeIdx/StepIdx）' if with_new else '旧协议（无 episode/step 字段）'))
print('=' * 70)
print(msg)
print('-- 序列化字节 (' + ('%d' % len(msg.SerializeToString())) + 'B) --')
print(msg.SerializeToString().hex())
'''


def run(label, pb2, extra=()):
    r = subprocess.run([_PY, "-c", _DUMP_SNIPPET, pb2, *extra],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{label}] 执行失败:\n{r.stderr}")
        return
    print(r.stdout)


if __name__ == "__main__":
    run("新", NEW_PB2, ("--with-new-fields",))
    run("旧", OLD_PB2)
