# OLAF SDK 与新版 rsl_rl 训练对齐报告

> 仓库：`python_olaf_sdk` ⟷ 新增 `rl/` 训练代码（Isaac Lab + rsl_rl）
> 报告日期：2026-04-28

## 一、训练侧关键事实（取自 `rl/` 文件夹）

| 项目 | 配置 / 数值 | 来源 |
|---|---|---|
| 任务 ID | `Olaf-Velocity-Flat-v0` / `Olaf-Velocity-Flat-Play-v0` | `tasks/manager_based/olaf_locomotion/__init__.py` |
| 机器人 | `OLAF_CFG`（URDF：`assets/URDF/olaf_robstride/urdf/olaf_robstride.urdf`） | `robots/olaf.py` |
| 自由度 | 12 个旋转关节（每腿 6 个），`fixed imu_joint` 通过 `merge_fixed_joints=True` 合并到 `base_link` | URDF + `olaf.py:43` |
| 默认姿态 `q₀` | `hip_yaw=0, hip_roll=0, hip_pitch=0.9, knee=1.65, ankle_pitch=0.7, ankle_roll=0`（左右同号） | `olaf.py:_BENT_KNEE_INIT` |
| 动作头 | `JointPositionActionCfg(scale=0.5, use_default_offset=True)` ⟹ `q_des = 0.5·a + q₀` | `olaf_env_cfg.py:75, 419` |
| 控制频率 | 50 Hz 策略，200 Hz 内环 PD（`sim.dt=0.005`，`decimation=4`） | base velocity_env_cfg |
| 观测维度 | **45**（顺序与缩放见下表） | `olaf_env_cfg.py:120-135` |
| 网络 | MLP `[512, 256, 128]`，激活 ELU；`obs_normalization=False` | `agents/rsl_rl_ppo_cfg.py` |
| 力矩刚度/阻尼 | RS02:40/3，RS03:78.957/5.027，RS00:16.581/1.056 | `olaf.py:75-90` |
| 转矩上限（仿真） | RS02 11.9 N·m，RS03 42 N·m，RS00 14 N·m | `olaf.py:effort_limit_sim` |
| 速度指令范围（训练） | `vx ∈ [-0.4, 0.7]`，`vy ∈ [-0.4, 0.4]`，`wz ∈ [-1.0, 1.0]` | `olaf_env_cfg.py:481-483` |
| 零指令处理 | 15% 环境以 `(0,0,0)` 起步；`bipedal_air_time/joint_position_penalty/GaitReward` 取消按指令幅度门控 ⟹ **零指令下原地踏步**（不是站定） | `olaf_env_cfg.py:57` + `mdp/rewards.py` |
| 终止条件 | `base_link / *_hip_*_link / *_knee_pitch_link` 接触地面 | `olaf_env_cfg.py:373` |
| 站立基座高度 | `base_height_l2.target_height = 0.30 m`（不是出生时的 0.45 m） | `olaf_env_cfg.py:278` |
| 期望摆腿高度 | `foot_clearance_reward.target_height = 0.10 m`（≈5.25 cm 离地） | `olaf_env_cfg.py:203` |
| IMU 安装（训练） | 挂在 `base_link` 上，`pos=(0,0,0)`、`rot=(1,0,0,0)`（即“IMU 帧 = base_link 帧”） | `olaf_env_cfg.py:409-415` |

### 观测向量布局（45 维，必须严格一致）

| idx | 项 | 缩放 | 维度 | 备注 |
|---:|---|---:|---:|---|
| 0:3 | `imu_ang_vel` | 0.20 | 3 | 含训练噪声 ±0.35（rad/s） |
| 3:6 | `imu_projected_gravity` | 1.00 | 3 | 单位向量 |
| 6:18 | `joint_pos_rel = q − q₀` | 1.00 | 12 | |
| 18:30 | `joint_vel × 0.05` | 0.05 | 12 | |
| 30:42 | `last_action`（**未缩放原始值**） | 1.00 | 12 | |
| 42:45 | `velocity_command (vx, vy, wz)` | 1.00 | 3 | |

---

## 二、与现有 SDK 的对齐审计

| 项目 | 训练侧 | SDK 当前实现 | 状态 |
|---|---|---|---|
| 观测顺序 / 缩放 | 见上表 | `observation.py` 顺序与缩放完全一致（`_ANG_VEL_SCALE=0.20`, `_JOINT_VEL_SCALE=0.05`） | ✅ 已对齐 |
| 默认关节姿态 `q₀` | 同左右、bent-knee | `config.DEFAULT_JOINT_POS` 与训练完全一致 | ✅ |
| 关节硬限位 | URDF | `config.JOINT_LIMITS` 与 URDF 完全一致 | ✅ |
| 内环 PD 增益 | RS02/03/00 上述值 | `config.MOTOR_TABLE` 数值一致 | ✅ |
| ONNX 输入 | `(1,45) float32` | `policy.py` 直接喂 raw obs；exporter 已在图中烘入 normalizer | ✅ |
| **动作映射 `q_des = 0.5·a + q₀`** | `scale=0.5` | `run.py:295`：`q_target = (action + DEFAULT_JOINT_POS)` — **缺少 ×0.5** | ❌ **严重错误** |
| 关节顺序 `JOINT_ORDER` | Isaac Lab `robot.data.joint_names` 按家族交替 [L,R] | `config.py:16-29` 与训练侧 12 维逐项匹配 | ✅ 已用训练机 `joint_names` 输出实测验证（2026-04-28） |
| 内环更新频率 | 200 Hz | `MOTOR_HZ = 200.0` ✅ | ✅ |
| IMU 帧 | base_link（训练侧 `ImuCfg.rot=(1,0,0,0)`） | `imu.py`: `_projected_gravity_from_quat` 改为 `-row₂(R)`；`R_base_from_imu` 改为 `np.eye(3)`（chip 实测与 base_link 对齐） | ✅ 已修复（2026-04-28） |
| 速度指令范围 | vx[-0.4,0.7], vy[-0.4,0.4], wz[-1,1] | `joystick.py` 已按 `VX_RANGE/VY_RANGE/WZ_RANGE` 逐轴 `np.clip` 到训练分布 | ✅ 已修复（2026-04-28） |
| 零指令行为 | 原地踏步（已去除指令门控） | README 与代码均未提示用户：“零摇杆位置≠静止站立” | ⚠️ 文档/期望需更新 |
| `soft_joint_pos_limit_factor=0.95` | 训练裁剪到 95% | `config.JOINT_LIMITS_SOFT`（95% 中心收缩），`run.py` 改用 `JOINT_LIMITS_SOFT` 裁剪 | ✅ 已修复（2026-04-28） |
| 转矩限幅 | RS02=11.9, RS03=42, RS00=14（训练仿真） | `MotorSpec.tau_limit` 灌入家族上限；`motors.command` 用 `τ̂ = kp·Δq + kd·Δqd` 推算并收缩 `q_des` | ✅ 已修复（2026-04-28） |
| `last_action` 寓义 | “raw, pre-scale” | `obs_builder.push_action(action)` 直接缓存 raw action | ✅ |
| `merge_fixed_joints` 影响 | imu_link 合并到 base_link | SDK 用 `IMU_OFFSET_IN_BASE` 做杠杆臂修正（仅对线速度有意义；ang_vel/proj_g 是刚体不变量） | ✅ 实际不影响策略 |
| 文档来源 | 旧 `SDK_DEPLOYMENT.md`（已被本次提交删除） | `observation.py` / `config.py` / `policy.py` 中的 `SDK_DEPLOYMENT.md §X` 引用全部清理，改为指向 `rl/.../olaf_env_cfg.py` 等真实代码位置 | ✅ 已修复（2026-04-28） |
| README 描述 | rsl_rl + ONNX | README 仍写 “skrl-trained walking policy” | ⚠️ 需修订 |

---

## 三、必须立即修复的问题（CRITICAL）

### 1. `run.py:295` 动作缩放缺失（最严重）— ✅ 已修复（2026-04-28）

训练侧 `JointPositionActionCfg(scale=0.5)` 表示

```
q_des = q₀ + 0.5 · action
```

修复后 SDK：

```python
# config.py 新增
ACTION_SCALE = 0.5  # 对齐 rl/.../olaf_env_cfg.py:75

# run.py:_policy_tick
action = self._policy(obs)
q_target = (ACTION_SCALE * action + DEFAULT_JOINT_POS).astype(np.float32)
```

### 2. 关节顺序 `JOINT_ORDER` 实测验证 — ✅ 已通过（2026-04-28）

训练机打印 `env.scene["robot"].data.joint_names` 输出：

```
['l_hip_yaw_joint', 'r_hip_yaw_joint',
 'l_hip_roll_joint', 'r_hip_roll_joint',
 'l_hip_pitch_joint', 'r_hip_pitch_joint',
 'l_knee_pitch_joint', 'r_knee_pitch_joint',
 'l_ankle_pitch_joint', 'r_ankle_pitch_joint',
 'l_ankle_roll_joint', 'r_ankle_roll_joint']
```

与 `config.JOINT_ORDER` 12 维逐项一致。**剩余的物理验证**：在物理机上用
`python run.py --slomo --joints l_hip_yaw`（依次替换名字）确认每个关节
名 ↔ CAN ID（`config.MOTOR_TABLE`）的映射也对——这是 SDK 侧最后一个
未在仿真里覆盖的环节。

---

## 四、建议同步更新的项目（HIGH）

### 3. 速度指令范围对齐训练分布 — ✅ 已修复（2026-04-28）

`joystick.py` 顶层新增训练命令包络常量并在 `_loop` 中逐轴 `np.clip`：

```python
VX_RANGE = (-0.4, 0.7)
VY_RANGE = (-0.4, 0.4)
WZ_RANGE = (-1.0, 1.0)
...
vx = float(np.clip(-ly * self._max_v, *VX_RANGE))
vy = float(np.clip(-lx * self._max_v, *VY_RANGE))
wz = float(np.clip(-rx * self._max_w, *WZ_RANGE))
```

后续若调整训练侧 `commands.base_velocity.ranges`，记得同步本三常量。

### 4. 关节目标软限位 — ✅ 已修复（2026-04-28）

`config.py` 在 `JOINT_LIMITS` 之后追加：

```python
SOFT_LIMIT_FACTOR = 0.95
_jl_mid = (JOINT_LIMITS[:, 0] + JOINT_LIMITS[:, 1]) * 0.5
_jl_half = (JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]) * 0.5 * SOFT_LIMIT_FACTOR
JOINT_LIMITS_SOFT = np.stack([_jl_mid - _jl_half, _jl_mid + _jl_half], axis=1).astype(np.float32)
```

`run.py:_policy_tick` 改为：

```python
q_target = np.clip(q_target, JOINT_LIMITS_SOFT[:, 0], JOINT_LIMITS_SOFT[:, 1])
```

硬限位 `JOINT_LIMITS` 仍保留为最后一道兜底常量，未来若需要可以二次封顶。

### 5. 软件转矩上限（保守安全门）— ✅ 已修复（2026-04-28）

训练 `effort_limit_sim`（11.9 / 42 / 14 N·m）比 URDF 标称（17 / 60 / 14 N·m）更紧。Robstride 走 MIT 模式由固件闭环，不能直接传"转矩限幅"参数，所以 SDK 改为**预估闭环转矩并收缩位置偏差**：

`config.py`：

```python
# Per-family torque caps (N·m) — mirror training effort_limit_sim
_RS02_TAU = 11.9   # hip_yaw
_RS03_TAU = 42.0   # hip_roll, hip_pitch, knee_pitch
_RS00_TAU = 14.0   # ankle_pitch, ankle_roll
# MotorSpec 新增 tau_limit 字段，全部 12 行 MOTOR_TABLE 都填上对应值。
```

`motors.command`：

```python
# τ̂ = kp·(q_des − q_meas) + kd·(qd_des − qd_meas)
# 阻尼项优先占用预算，剩下的预算决定允许的 (q_des − q_meas)。
tau_kd = kd * (qd_cmd - qd_meas)
budget = s.tau_limit - abs(tau_kd)
if budget <= 0.0:
    q_cmd = q_meas
else:
    max_dq = budget / kp
    q_cmd = clip(q_cmd, q_meas - max_dq, q_meas + max_dq)
```

这样无论策略瞬间想要多大幅度的位置目标，闭环输出都不会超出训练时见过的力矩包络。Robstride 固件层若另有上限，本封顶相当于第二道兜底。

### 6. 零指令“原地踏步”是预期行为，必须更新文档与心理模型

新训练取消了 `bipedal_air_time_reward`、`joint_position_penalty`、`GaitReward` 三处的“指令幅度门控”。手柄回中时机器人会**继续抬腿踏步**，不是静立。

- README “Joystick control” 一节应加注：*“摇杆回中时机器人持续踏步，属正常行为；按 X 切到零位（或重启）才会真正静立”*。
- `--debug` 调试模式的 10 步动作策略也会在 cmd=0 下抬腿，需操作员预期。

### 7. 悬空 `SDK_DEPLOYMENT.md` 引用清理 — ✅ 已修复（2026-04-28）

`observation.py` / `config.py` / `policy.py` 中所有 `SDK_DEPLOYMENT.md §X` 引用已替换为真实代码定位（`rl/olaf_bipedal_robot/.../olaf_env_cfg.py`、`olaf.py` 等），并补全 ONNX normalizer 的处境说明（训练侧 `obs_normalization=False`，所以 baked-in normalizer 当前是 identity）。

如未来需要更详尽的部署手册，可参考 git 历史 `9d0e790:SDK_DEPLOYMENT.md` 重建一份，但这是可选项，不阻塞部署。

### 8. 修正 README

`README.md` 第 4 行写 “skrl-trained” 应改为 “rsl_rl-trained”；“Workflow” 中只有 `uv sync && python run.py` 远不够；建议把“训练 → 导出 → 部署”三步显式化：

```bash
# 训练 (Isaac Lab + rsl_rl, 训练机)
cd rl
isaaclab.bat -p scripts/rsl_rl/train.py --task Olaf-Velocity-Flat-v0
# 导出 (训练机)
isaaclab.bat -p scripts/rsl_rl/export_policy.py --checkpoint <path>
# 部署 (机器人)
cp <exported>/policy.onnx ./policy.onnx
python run.py
```

### 9. IMU 帧与 projected_gravity 公式 — ✅ 已修复（2026-04-28）

实测 dump（机器人静止水平）显示：

```
accel = (0.59, 0.65, -9.79)  m/s²
q     = (-0.5294, -0.0074, -0.0440, 0.8472)
Roll  = 0.0592 rad,  Pitch = 0.0668 rad,  Heading = 4.2566 rad
```

得出两个结论：

1. **chip_z 与 base_z 同向（向上）**：`accel_z ≈ -g` 且 `R[2,2] ≈ +0.996` 共同说明 chip 是"正装"，与 base_link 的 REP-103 约定（forward=+x, left=+y, up=+z）一致。
   - 把原 `R_base_from_imu = diag(-1, 1, -1)` 改成 `np.eye(3)`。
   - 之前的 `diag(-1, 1, -1)` 实际把 `proj_g_z` 翻成了 +1，等于让策略以为重力指向上，启动 `B` 必摔。

2. **`_projected_gravity_from_quat` 公式错的是第 3 列 vs 第 3 行**：
   原代码计算 `-col₂(R) = -(2(xz+wy), 2(yz-wx), 1-2(x²+y²))`，但对 body-to-world 的 `R`，body 帧重力应是 `R^T @ (0,0,-1) = -row₂(R) = -(2(xz-wy), 2(yz+wx), 1-2(x²+y²))`。Z 分量恰好是对角元，所以"看起来差不多对"，但只要存在 roll/pitch，X/Y 就完全错位。

修复后用 dump 帧自检：

```
chip-frame proj_g  = ( 0.0591,  0.0667, -0.9960)
base-frame ang_vel = (-0.002 , -0.0021,  0.0003)   # 静止 ✓
base-frame proj_g  = ( 0.0591,  0.0667, -0.9960)   # |·|=1.0000, z≈-1 ✓
```

**还需要的物理 yaw 验证**（写进 P3 自检）：

| 操作 | 期望 |
|---|---|
| 静置水平 | `proj_g_z ≈ -1`，XY ≈ 0 |
| 前倾 ~10° | `proj_g_x ≈ +0.17`，`proj_g_y ≈ 0` |
| 左倾 ~10° | `proj_g_y ≈ -0.17`，`proj_g_x ≈ 0` |
| 绕 z 顺时针 | `ang_vel_z` 为负（CCW 正） |

如果"前倾时变化的是 `proj_g_y` 而不是 `proj_g_x`"，说明 chip 在水平面里多绕了 90°，那就给 `R_base_from_imu` 加一个**只动 z 平面、不动 z 行**的 90° 倍数子矩阵（如 `[[0,-1,0],[1,0,0],[0,0,1]]`）。**绝不要再动 z 行的符号**——已被实测 dump 钉死。

---

## 五、建议增补的对齐自检（验收清单）

部署到物理机前建议在 `tools/` 加一个 `sdk_smoke_test.py`（不到 100 行即可），完成：

1. **维度自检** — `OBS_DIM == 45`、`onnx output shape == (1, 12)`。
2. **零输入测试** — 喂 `np.zeros((1,45))` 给 ONNX，断言 `|action| < 1` 每维。
3. **关节顺序回环** — 用零位 + 单一非零 `joint_pos_rel[k]` 做正向验证：策略输出应主要影响第 `k` 关节。
4. **同模型 sim/real 对比** — 把训练机 `play.py` 的同一帧 obs（取 PLAY 任务下一步）保存为 `.npy`，物理机加载它，比较两边 `policy(obs)` 输出 ≤ 1e-5 差。如不一致说明 ONNX 导出或 obs 构造有问题。
5. **boot ramp 时序** — 验证 `MOVE_DEFAULT` (Y) 后 `_pose_tolerance=0.15 rad` 真的能在 ~3 s 达到，且 `_kp_scale` 已 ramp 到 1.0。
6. **IMU 倾斜测试**（4 步，确认 `R_base_from_imu` 是否还需要 yaw 子转）：
   - 静置水平：`proj_g_z ≈ -1`，XY ≈ 0
   - 前倾 ~10°：`proj_g_x ≈ +0.17`，`proj_g_y ≈ 0`
   - 左倾 ~10°：`proj_g_y ≈ -0.17`，`proj_g_x ≈ 0`
   - 绕 z 顺时针自转：`ang_vel_z` 为负

---

## 六、推荐修改的代码清单（按优先级）

| 优先级 | 文件 | 修改 |
|---|---|---|
| ~~P0~~ ✅ | `run.py:296` | `q_target = ACTION_SCALE * action + DEFAULT_JOINT_POS` — 已修复 |
| ~~P0~~ ✅ | `config.py` | 新增 `ACTION_SCALE = 0.5` 常量 — 已新增 |
| ~~P0~~ ✅ | 训练机 + `config.py:JOINT_ORDER` | 用训练机 `joint_names` 实测顺序 — 已验证一致 |
| ~~P1~~ ✅ | `joystick.py` | vx/vy/wz 已逐轴裁剪到训练范围 |
| ~~P1~~ ✅ | `config.py` + `run.py` | `JOINT_LIMITS_SOFT`（×0.95）已添加并接入 |
| ~~P1~~ ✅ | `motors.py:command` | 软件转矩封顶 11.9/42/14 N·m，已通过 `τ̂ = kp·Δq + kd·Δqd` 反推收缩 q_des |
| P2 | `README.md` | “skrl-trained” → “rsl_rl-trained”；补充训练-导出-部署流程 |
| ~~P2~~ ✅ | `observation.py / config.py / policy.py` | 已删除全部 `SDK_DEPLOYMENT.md §X` 悬空引用 |
| ~~P0~~ ✅ | `imu.py` | `R_base_from_imu` 改为 `np.eye(3)`；`_projected_gravity_from_quat` 修正为 `-row₂(R)`，dump 自检通过 |
| P2 | 文档 | 注明 “零指令 ⇒ 原地踏步” 为预期 |
| P3 | `tools/sdk_smoke_test.py` | 新增第 5 节列出的 5 项启动自检（含 IMU 倾斜验证 4 项） |

---

## 七、与训练侧需要保持的“硬契约”摘要

| 契约 | 数值 | 一旦改动需同步的地方 |
|---|---|---|
| 观测维度 | 45 | `observation.py:OBS_DIM`，`policy.py` 输入 shape |
| 动作维度 | 12 | `config.N_JOINTS` |
| 动作缩放 | 0.5 | `config.ACTION_SCALE`（建议新增） |
| 默认姿态 `q₀` | 见 `olaf.py:_BENT_KNEE_INIT` | `config.DEFAULT_JOINT_POS` |
| 关节顺序 | Isaac Lab 实际枚举 | `config.JOINT_ORDER` |
| 控制频率 | 50 Hz 外环，200 Hz 内环 | `config.POLICY_HZ`, `MOTOR_HZ` |
| IMU 帧 | base_link | `imu.R_base_from_imu` |
| 速度指令范围 | (-0.4,0.7)/(-0.4,0.4)/(-1,1) | `joystick.py` 裁剪 |
| ONNX 输入 | `obs (1,45)` raw | `policy.py`（已对齐） |

> 训练侧任何下次修改 `olaf_env_cfg.ObservationsCfg.PolicyCfg` 顺序 / 缩放、`ActionsCfg.scale`、`OLAF_CFG._BENT_KNEE_INIT`、URDF 关节限位时，必须在同一 commit 同步本表对应 SDK 字段，否则策略行为会“看似工作但慢慢漂移”，不可接受。

---

## 八、Raspberry Pi 5（4 GB）部署性能评估（2026-04-29）

> 评估目标：本 SDK 在 Pi 5 / 4 GB 上能否实时驱动 12-DOF 双足策略（50 Hz 外环、200 Hz 内环、AHRS @ ≥ 100 Hz）。
>
> 结论：**可以，且有充足余量**。瓶颈不在 CPU/RAM，而在 **CAN 接口选型** 与 **Linux 实时性**。

### 8.1 计算预算

| 循环 | 频率 | 周期 | 来源 |
|---|---:|---:|---|
| ONNX 策略 + obs build (`run.py:_policy_tick`) | 50 Hz | 20 ms | `config.POLICY_HZ` |
| 电机命令 + LPF + 反馈泵 (`_motor_loop`) | 200 Hz | 5 ms | `config.MOTOR_HZ` |
| IMU 读取线程（AHRS） | 100–400 Hz | 2.5–10 ms | `imu.py:_poll_loop` |
| 手柄轮询 | 100 Hz | 10 ms | `joystick.py:_loop` |

### 8.2 策略推理成本

第一节给出网络结构 MLP `[512, 256, 128]`，ELU，输入 45 → 输出 12，**约 188 k MAC / 推理**：

```
45×512 + 512×256 + 256×128 + 128×12
= 23,040 + 131,072 + 32,768 + 1,536
≈ 188,416 MACs
```

Pi 5（Cortex-A76 @ 2.4 GHz，NEON）跑 ONNX Runtime CPU EP，FP32，`intra_op_num_threads=1`，单次推理 **0.2–0.5 ms**。50 Hz 预算 20 ms，占用 **<3 % 单核**。模型太小，GPU/NPU 反而被调用开销吃掉收益，**坚持 CPU EP**。

### 8.3 电机线程成本

`_motor_loop` 每 5 ms 做：

1. LPF（12 个 float） — 亚 µs。
2. `motors.command` → 12 × `write_operation_frame` → 12 × `socketcan send`，每次约 30 µs，合计 **~0.4 ms**。
3. `pump_feedback(budget_s=0.0005)` — 非阻塞耗排，已有上限。

每 tick 总耗时 **~1 ms**，5 ms 预算余量充足。

### 8.4 CAN 总线选型（关键风险）

`config.py:165-167` 已注明：12 motors × 200 Hz × (cmd + reply) 大约 **4800 帧/s 总量**。被 `can_usb`/`can_spi` 拆成左右腿各半。

| 接口 | 评价 |
|---|---|
| **USB-CAN（CANable / CANUSB）** @ 1 Mbps | 单条 2400 帧/s 完全 OK，socketcan 送得动 ≥ 50k/s。**推荐**。 |
| **MCP2515 on SPI**（Pi 5 默认 HAT 常用芯片） | 单条 2400 帧/s **逼近上限**（MCP2515 在 ~700 帧/s 以上突发会丢帧）。**不推荐**作为 `can_spi`。 |
| **MCP2518FD on SPI** | CAN-FD 控制器，Pi 5 高速 SPI 完全够用，可作为 `can_spi`。 |
| 双通道 USB-CAN-FD | 最干净的方案，两路都用 USB。 |

> **行动项**：若当前 Pi 5 用了 MCP2515 HAT 做 `can_spi`，切到 **双 USB-CAN** 或 **MCP2518FD**。这是部署到物理机上需要先确认的硬件项。

CAN-FD（`config.CAN_FD_ENABLED`）目前关闭。开启后 5 Mbps 数据相需要：① 内核 `fd on`，② 每个电机固件配 FD，③ 适配器支持 FD。前两条没做之前不要打开。

### 8.5 IMU 串口（921600 baud）

Pi 5 PL011 硬 UART 跑 921600 完全稳；CP2102 USB-UART（`imu.py:50-53` 路径）走 USB CDC-ACM，波特率不影响实际带宽。**无问题**。

切忌把 IMU 降到 115200——AHRS 帧 56 字节 + IMU 帧 64 字节 = 一对 1200 bit，100 Hz 已经超过 115200 ceil，会丢帧 → policy 拿到陈旧 `proj_g` → 不可接受（详见 §九 IMU 帧契约）。

### 8.6 内存

冷启动占用：

| 组件 | 占用 |
|---|---:|
| Python + numpy + pyserial + python-can + onnxruntime | ~250 MB |
| ONNX session（小 MLP） | ~15 MB |
| pygame（手柄） | ~30 MB |
| 机器人状态 / obs 历史 / LPF 状态 | < 1 MB |
| **总计** | **~300 MB** |

4 GB 大量过剩，2 GB Pi 5 也能跑。冗余可用于额外的 rosbag 日志、Web 仪表盘等。

### 8.7 实时性（真正的风险）

Linux 不是硬实时系统。两个具体风险：

1. **`POLICY_WATCHDOG_S = 0.040`**（`config.py:177`）：单次策略 tick > 40 ms 时，电机线程会切到 damping。stock Raspberry Pi OS 会偶发 10–30 ms 的调度卡顿（GC、磁盘 I/O 抢占、Swap）。
2. **GIL**：ONNX `Run()` 内部释放 GIL，策略推理期间电机线程能跑；但 `observation.py` obs-build 是纯 Python，会持 GIL。三线程（policy / motor / IMU）在 Pi 5 四核上不冲突，但若再加高频日志线程要小心。

#### 推荐 Pi 5 上的加固清单

```bash
# 1. 关闭 swap（避免 stop-the-world）
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile

# 2. 隔离 CPU（在 /boot/firmware/cmdline.txt 加）
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3

# 3. 用 SCHED_FIFO 跑控制线程（在启动脚本里）
sudo chrt -f 90 -p $POLICY_TID
sudo chrt -f 80 -p $MOTOR_TID
taskset -cp 2 $POLICY_TID
taskset -cp 3 $MOTOR_TID

# 4. 可选：装 PREEMPT_RT 内核（最坏延迟从 ~10 ms 降到 <500 µs）
#    仅当你看到 "policy tick %.1f ms > watchdog" 警告时再考虑
```

观察手段：`run.py` 已内建 `policy tick %.1f ms > watchdog`（`run.py` 中 `_tick` 的检测）。部署后第一周关注该 warning 频率，作为是否上 PREEMPT_RT 的依据。

### 8.8 部署前 Pi 5 验收清单

| # | 检查项 | 通过判据 |
|---|---|---|
| 1 | CAN 接口选型 | 两路均为 USB-CAN-FD 或 USB-CAN 1 Mbps；MCP2515 不应出现在 `can_spi` |
| 2 | `config.CAN_FD_ENABLED` 与硬件一致 | False（默认）或 True 但内核 + 电机固件 + 适配器都 FD-ready |
| 3 | IMU 波特率 | `imu.DEFAULT_BAUD = 921_600`，不要降 |
| 4 | Swap 关闭 | `swapon --show` 无输出 |
| 5 | CPU 隔离 | `cat /sys/devices/system/cpu/isolated` 显示 `2-3` 或类似 |
| 6 | 控制线程 SCHED_FIFO | `chrt -p $TID` 显示 `policy: SCHED_FIFO` |
| 7 | 单次策略推理时长 | 持续观察 5 分钟，无 `policy tick ... > watchdog` 警告 |
| 8 | CAN 帧丢失 | `ip -s -d link show can_usb` 的 `dropped/overrun` 长期为 0 |

### 8.9 与训练侧的接口稳定性

Pi 5 部署不影响 §七 中列出的“硬契约”——观测维度 / 缩放 / 关节顺序 / 控制频率 / IMU 帧 等都来自训练侧固定常量，与部署平台无关。本节增补的是**平台侧的非功能要求**，目的是让训练侧的契约在 Pi 5 上能稳定满足。