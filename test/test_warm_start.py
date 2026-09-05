"""IL → RL 热启动（方案 A）验证 —— 只迁移主干，双头保持随机。

覆盖四条路径:
    1. 正常迁移   IL 单头 DRQN 权重 → RL 双头 DualHeadDRQN，主干逐字节一致、
                  双头保持随机、target 网络已同步
    2. 维度失配   IL 与 RL 跑在不同场景（input_shape 不同）时，报出人话错误
    3. 前缀选取   目录内有多份权重时，优先取与当前场景前缀一致的那份
    4. 文件缺失   目录无权重时报 FileNotFoundError

运行::

    python test/test_warm_start.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import torch

from services.algorithms.qmix.nn import DRQN
from services.algorithms.qmix.policy import _BasePolicy
from utils.weight_naming import build_weight_prefix, weight_file_name

# scene.json 的真实规模：6 雷达 / 0 卫星 / 4 目标
N_RADARS, N_SATELLITES, N_TARGETS = 6, 0, 4


def make_conf(n_radars=N_RADARS, n_satellites=N_SATELLITES, n_targets=N_TARGETS):
    """构造与 scene.json 同规模的最小 conf（IL 与 RL 共用同一套公式）。"""
    n_agents = n_radars + n_satellites
    n_actions = n_targets + 1
    obs_dim = 9 + n_targets * (5 + 3)      # RADAR_SELF + n_targets*(TARGET+SAT_BROADCAST)
    return SimpleNamespace(
        device='cpu',
        drqn_hidden_dim=64,
        obs_shape=obs_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        n_ld=n_radars,
        n_wx=n_satellites,
        n_targets=n_targets,
        ld_n_actions=n_targets,
        n_radars=n_radars,
        n_satellites=n_satellites,
        group_size=0,
        last_action=True,
        reuse_network=True,
        model_dir=None,
        learning_rate=5e-4,
    )


def il_input_shape(conf):
    obs_dim = conf.obs_shape[0] if isinstance(conf.obs_shape, tuple) else conf.obs_shape
    return obs_dim + conf.n_actions + conf.n_agents


def save_il_weight(conf, out_dir, seed=1234):
    """按 ILAgents.save_model 的规则落一份单头 DRQN 权重。"""
    torch.manual_seed(seed)
    net = DRQN(il_input_shape(conf), conf)
    prefix = build_weight_prefix(
        conf.n_radars, conf.n_satellites, conf.n_targets, conf.group_size,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, weight_file_name(prefix, 'drqn'))
    torch.save(net.state_dict(), path)
    return net, path


# ============================================================

def test_normal_transfer(tmp):
    print("\n" + "=" * 64)
    print("[用例1] 正常迁移：主干搬过去，双头保持随机")
    print("=" * 64)

    conf = make_conf()
    load_dir = os.path.join(tmp, "il_weights")
    il_net, _ = save_il_weight(conf, load_dir)

    policy = _BasePolicy(conf, dual_head=True)
    before = {k: v.clone() for k, v in policy.eval_drqn_net.state_dict().items()}

    policy.load_drqn_state(load_dir)
    after = policy.eval_drqn_net.state_dict()
    il_state = il_net.state_dict()

    # 主干必须与 IL 权重逐字节一致
    for key in il_state:
        if key.startswith(_BasePolicy.TRUNK_PREFIXES):
            assert torch.equal(after[key], il_state[key]), f"主干 {key} 未正确迁移"
    print(f"  [OK] 主干 8 个张量与 IL 权重逐字节一致")

    # 双头必须保持随机初始化（未被触碰）
    for key in ('fc_ld.weight', 'fc_ld.bias', 'fc_wx.weight', 'fc_wx.bias'):
        assert torch.equal(after[key], before[key]), f"{key} 不应被修改"
    print(f"  [OK] fc_ld / fc_wx 保持随机初始化，未被 IL 权重污染")

    # IL 的 fc2 不得出现在 RL 网络里
    assert 'fc2.weight' not in after, "IL 的 fc2 泄漏进了 RL 网络"
    print(f"  [OK] IL 的 fc2 已丢弃")

    # target 网络必须同步，否则 TD target 用的是随机主干
    for key, val in after.items():
        assert torch.equal(policy.target_drqn_net.state_dict()[key], val), \
            f"target 网络 {key} 未同步"
    print(f"  [OK] target_drqn_net 已与 eval 同步")


def test_shape_mismatch(tmp):
    print("\n" + "=" * 64)
    print("[用例2] 维度失配：IL 与 RL 跑在不同场景")
    print("=" * 64)

    # IL 在 6/0/4 小场景训练
    il_conf = make_conf(6, 0, 4)
    load_dir = os.path.join(tmp, "small_scene")
    save_il_weight(il_conf, load_dir)

    # RL 却跑 100/25/21 生产场景
    rl_conf = make_conf(100, 25, 21)
    policy = _BasePolicy(rl_conf, dual_head=True)

    try:
        policy.load_drqn_state(load_dir)
    except ValueError as e:
        msg = str(e)
        assert "不匹配" in msg and "input_shape" in msg, "报错缺少关键诊断信息"
        print("  [OK] 抛出可读报错：")
        for line in msg.splitlines():
            print(f"       {line}")
        return
    raise AssertionError("维度失配却未报错")


def test_prefix_preference(tmp):
    print("\n" + "=" * 64)
    print("[用例3] 多份权重：优先取与当前场景前缀一致的那份")
    print("=" * 64)

    load_dir = os.path.join(tmp, "mixed")

    # 先落一份别的场景（同 input_shape 但前缀不同），再落本场景
    other = make_conf(3, 3, 4)       # 3+3=6 agents, 4 targets -> input_shape 同为 52
    save_il_weight(other, load_dir, seed=999)
    conf = make_conf()
    il_net, want_path = save_il_weight(conf, load_dir, seed=1234)

    names = sorted(os.listdir(load_dir))
    assert len(names) == 2, f"预期 2 份权重，实际 {names}"
    print(f"  目录内权重: {names}")

    policy = _BasePolicy(conf, dual_head=True)
    policy.load_drqn_state(load_dir)

    got = policy.eval_drqn_net.state_dict()['fc1.weight']
    assert torch.equal(got, il_net.state_dict()['fc1.weight']), \
        "取到了错误场景的权重"
    print(f"  [OK] 精确命中 {os.path.basename(want_path)}，未误取另一份")


def test_missing_file(tmp):
    print("\n" + "=" * 64)
    print("[用例4] 权重缺失")
    print("=" * 64)

    empty = os.path.join(tmp, "empty")
    os.makedirs(empty, exist_ok=True)
    policy = _BasePolicy(make_conf(), dual_head=True)
    try:
        policy.load_drqn_state(empty)
    except FileNotFoundError as e:
        print(f"  [OK] {e}")
        return
    raise AssertionError("权重缺失却未报错")


def test_lr_groups_pure_rl(tmp):
    """纯 RL 从零训练：两个参数组 lr 必须相同 = 与不分组时行为等价。"""
    print("\n" + "=" * 64)
    print("[用例5] 纯 RL 从零训练：分组不改变学习率")
    print("=" * 64)

    conf = make_conf()
    policy = _BasePolicy(conf, dual_head=True)
    head = list(policy.eval_drqn_net.fc_ld.parameters())
    opt, flat = policy._make_optimizer(head)

    lrs = [g['lr'] for g in opt.param_groups]
    assert len(lrs) == 2, f"预期 2 个参数组，实际 {len(lrs)}"
    assert lrs[0] == lrs[1] == conf.learning_rate, \
        f"从零训练时两组 lr 应相同且等于 base_lr，实际 {lrs}"
    print(f"  [OK] 两个参数组 lr 均为 {conf.learning_rate:.2e}（与不分组等价）")

    # 主干必须在 TRUNK_GROUP 下标上，否则 set_trunk_lr_scale 会调错组
    trunk_ids = {id(p) for p in policy.trunk_parameters()}
    group0_ids = {id(p) for p in opt.param_groups[_BasePolicy.TRUNK_GROUP]['params']}
    assert trunk_ids == group0_ids, "主干不在 TRUNK_GROUP 下标上"
    print(f"  [OK] 主干位于 param_groups[{_BasePolicy.TRUNK_GROUP}]")

    # 扁平列表供 clip_grad_norm_ 使用，必须无遗漏无重复
    assert len(flat) == len(policy.trunk_parameters()) + len(head), "扁平列表数量不符"
    assert len({id(p) for p in flat}) == len(flat), "扁平列表存在重复参数"
    print(f"  [OK] 扁平参数列表 {len(flat)} 项，无重复（供 clip_grad_norm_ 使用）")


def test_lr_scale_after_warm_start(tmp):
    """热启动成功后：主干组降为 0.1×，头组保持原速。"""
    print("\n" + "=" * 64)
    print("[用例6] 热启动后：主干单独降速")
    print("=" * 64)

    conf = make_conf()
    load_dir = os.path.join(tmp, "lr_scale")
    save_il_weight(conf, load_dir)

    policy = _BasePolicy(conf, dual_head=True)
    policy._make_optimizer(list(policy.eval_drqn_net.fc_ld.parameters()))
    policy.load_drqn_state(load_dir)

    groups = policy._trunk_optimizer.param_groups
    base = conf.learning_rate
    expected = base * _BasePolicy.DEFAULT_WARM_START_TRUNK_LR_SCALE
    assert abs(groups[0]['lr'] - expected) < 1e-12, \
        f"主干 lr 应为 {expected:.2e}，实际 {groups[0]['lr']:.2e}"
    assert abs(groups[1]['lr'] - base) < 1e-12, \
        f"头组 lr 应保持 {base:.2e}，实际 {groups[1]['lr']:.2e}"
    print(f"  [OK] 主干 {groups[0]['lr']:.2e} (0.1×) / 头组 {groups[1]['lr']:.2e} (原速)")

    # scale >= 1.0 视为不启用，不应改动
    policy.set_trunk_lr_scale(1.0)
    assert abs(groups[0]['lr'] - expected) < 1e-12, "scale=1.0 不应改动 lr"
    print(f"  [OK] set_trunk_lr_scale(1.0) 视为不启用，未改动")


def test_grouping_numerically_equivalent(tmp):
    """数值等价：同 lr 下，分组优化器与扁平优化器的一步更新结果完全一致。"""
    print("\n" + "=" * 64)
    print("[用例7] 数值等价：分组 vs 不分组，一步更新逐字节相同")
    print("=" * 64)

    conf = make_conf()
    inp = il_input_shape(conf)

    def run(grouped):
        torch.manual_seed(20260826)
        net = DRQN(inp, conf)
        trunk = (list(net.fc1.parameters())
                 + list(net.layer_norm.parameters())
                 + list(net.rnn.parameters()))
        head = list(net.fc2.parameters())
        if grouped:
            opt = torch.optim.RMSprop([
                {'params': trunk, 'lr': conf.learning_rate},
                {'params': head, 'lr': conf.learning_rate},
            ])
        else:
            opt = torch.optim.RMSprop(trunk + head, lr=conf.learning_rate)

        torch.manual_seed(7)
        obs = torch.randn(8, inp)
        h = torch.zeros(8, conf.drqn_hidden_dim)
        for _ in range(3):                      # 多步，确保 RMSprop 状态也一致
            q, _h = net(obs, h)
            loss = (q ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net.state_dict()

    a, b = run(grouped=False), run(grouped=True)
    for key in a:
        assert torch.equal(a[key], b[key]), f"参数 {key} 在分组前后不一致"
    print(f"  [OK] 3 步更新后 {len(a)} 个张量逐字节相同 —— 纯 RL 路径零影响")


def main():
    tmp = tempfile.mkdtemp(prefix="warmstart_")
    try:
        test_normal_transfer(tmp)
        test_shape_mismatch(tmp)
        test_prefix_preference(tmp)
        test_missing_file(tmp)
        test_lr_groups_pure_rl(tmp)
        test_lr_scale_after_warm_start(tmp)
        test_grouping_numerically_equivalent(tmp)
        print("\n" + "=" * 64)
        print("[结论] 方案 A + 主干差分学习率，七条路径全部通过")
        print("=" * 64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
