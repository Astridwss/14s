"""验证 proto 加字段后线协议兼容性（旧/新 pb2 字节级对比）。

分进程各自加载一个 pb2（两个模块同名 'ProtoStruct.proto'，同进程加载会撞默认 descriptor pool），
把「只填老字段」的参考消息序列化后比对字节；再交叉验证：
  - 旧协议能否解析带新字段的帧（应正常读老字段、忽略未知字段）
  - 新协议能否解析旧帧（新字段应取默认 0）

用法（仓库根目录）::

    .venv/Scripts/python.exe scripts/verify_wire_compat.py
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_PB2 = os.path.join(_ROOT, "services", "zmq", "proto", "ProtoStruct_pb2.py.bak")
NEW_PB2 = os.path.join(_ROOT, "services", "zmq", "proto", "ProtoStruct_pb2.py")
_PY = sys.executable

_BUILD_SNIPPET = r'''
import sys
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('pb2mod', sys.argv[1]).load_module()
m = mod.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
m.CurrentTime = 12.5
m.DataType = 1
m.PubCtrl.MsgHeader.MsgType = 12290
e = m.EquipPos.add(); e.EquipID=101; e.EquipType=1; e.Time=12.5; e.GeoPos.X=1; e.GeoPos.Y=2; e.GeoPos.Z=3
t = m.TargetPos.add(); t.TargetID=7; t.Time=12.5; t.GeoPos.X=3; t.GeoPos.Y=4; t.GeoPos.Z=5
d = m.Detection.add(); d.EquipID=101; d.TargetID=7
if '--with-new-fields' in sys.argv:
    m.EpisodeIdx = 7
    m.StepIdx = 33
sys.stdout.write(m.SerializeToString().hex())
'''

_PARSE_SNIPPET = r'''
import sys
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('pb2mod', sys.argv[1]).load_module()
raw = open(sys.argv[2], 'rb').read()
m = mod.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
m.ParseFromString(raw)
ep = getattr(m, 'EpisodeIdx', 0)
sp = getattr(m, 'StepIdx', 0)
print('CurrentTime=%s DataType=%s EquipPos=%d TargetPos=%d Detection=%d EpisodeIdx=%s StepIdx=%s' % (
    m.CurrentTime, m.DataType, len(m.EquipPos), len(m.TargetPos), len(m.Detection), ep, sp))
'''


def _run(code: str, *args):
    return subprocess.run([_PY, "-c", code, *args], capture_output=True, text=True)


def main():
    # 1) 老字段字节一致性：旧/新各串化一份只填老字段的消息
    old_hex = _run(_BUILD_SNIPPET, OLD_PB2).stdout.strip()
    new_hex = _run(_BUILD_SNIPPET, NEW_PB2).stdout.strip()
    same = old_hex == new_hex
    print(f"[1] 老字段线协议字节一致: {same}  (old={len(old_hex)//2}B new={len(new_hex)//2}B)")

    # 2) 旧协议解析「带新字段」的帧：新 pb2 串化含 EpisodeIdx/StepIdx 的帧落盘，旧 pb2 解析
    with_new = _run(_BUILD_SNIPPET, NEW_PB2, "--with-new-fields").stdout.strip()
    bin_path = os.path.join(_ROOT, "scripts", "_wire_new.bin")
    with open(bin_path, "wb") as f:
        f.write(bytes.fromhex(with_new))
    r = _run(_PARSE_SNIPPET, OLD_PB2, bin_path).stdout.strip()
    print(f"[2] 旧协议解析带新字段帧: {r}")

    # 3) 新协议解析旧帧：旧 pb2 串化落盘，新 pb2 解析
    old_bin = os.path.join(_ROOT, "scripts", "_wire_old.bin")
    with open(old_bin, "wb") as f:
        f.write(bytes.fromhex(old_hex))
    r = _run(_PARSE_SNIPPET, NEW_PB2, old_bin).stdout.strip()
    print(f"[3] 新协议解析旧帧: {r}")

    for p in (bin_path, old_bin):
        if os.path.exists(p):
            os.remove(p)

    print("PASS" if (same and "EpisodeIdx=0" in r) else "FAIL")


if __name__ == "__main__":
    main()
