# Dreambo Python SDK — 从零到部署完整操作手册

本仓库是 OLAF 双足机器人步态策略的 **运行时 SDK**（Runtime SDK），负责把 skrl / rsl_rl 在 Isaac Lab 中训练得到的 ONNX 步态网络部署到真实硬件：在树莓派（或同等 Linux 单板机）上以 50 Hz 跑策略推理、200 Hz 下发 CAN 指令到电机，并通过手柄完成开关机/急停/姿态切换。

> 本手册面向 **第一次接触本项目的工程师**，覆盖从硬件清单、依赖安装、CAN/IMU/手柄配置、首次上电、关节零点标定，到最终用策略走起来的全部步骤。每一步都给出可复制粘贴的命令。

---

## 目录

1. [项目简介](#1-项目简介)
2. [硬件清单](#2-硬件清单)
3. [软件环境准备](#3-软件环境准备)
4. [拉取与依赖安装](#4-拉取与依赖安装)
5. [硬件接口配置](#5-硬件接口配置)
6. [关键参数核对（**上电前必做**）](#6-关键参数核对上电前必做)
7. [首次上电流程](#7-首次上电流程)
8. [关节零点标定](#8-关节零点标定)
9. [手柄使用与按键说明](#9-手柄使用与按键说明)
10. [调试与单关节验证](#10-调试与单关节验证)
11. [安全机制](#11-安全机制)
12. [常见问题排查](#12-常见问题排查)
13. [文件结构速查](#13-文件结构速查)

---

## 1. 项目简介

| 模块             | 作用                                                                 |
|------------------|----------------------------------------------------------------------|
| `config.py`      | 关节顺序、默认姿态、电机型号、CAN ID、关节限位、回路频率等所有常量   |
| `can_bus.py`     | 基于 `python-can` 的 SocketCAN 薄封装                                |
| `motors.py`      | Robstride RS00 / RS02 / RS03 与达妙 DM-J43xx MIT 模式驱动            |
| `imu.py`         | HI226 / HI229 串口 AHRS 模块解析（独立线程）                          |
| `observation.py` | 观测向量拼装 + path-frame 跟踪器                                     |
| `policy.py`      | ONNX 策略推理 + RunningStandardScaler 复现                           |
| `joystick.py`    | Xbox One S 手柄驱动（pygame / SDL）                                  |
| `run.py`         | 主入口：50 Hz 策略循环 + 200 Hz 电机循环 + 状态机                    |

**控制频率：**
- 策略：`POLICY_HZ = 50`
- 电机：`MOTOR_HZ = 200`
- 看门狗：策略 tick 超过 40 ms 自动进入阻尼模式
- 启动 kp 软斜坡：2 秒（`KP_RAMP_S`）

---

## 2. 硬件清单

| 类别        | 推荐型号 / 规格                                              |
|-------------|--------------------------------------------------------------|
| 主控        | Raspberry Pi 4B / 5（或任意 ARM64 Linux 单板机）             |
| CAN 适配器  | CANable v2 / candleLight 等 SocketCAN 兼容设备，1 Mbps       |
| 电机        | Robstride RS02（髋偏航）× 2，RS03（髋滚转/俯仰、膝）× 6，RS00（踝俯仰/滚转）× 4 |
| IMU         | HI226 / HI229（CP2102 USB-UART，921600 bps）                 |
| 手柄        | Xbox One S 控制器（USB 或蓝牙，**必备**）                    |
| 电源        | 与电机额定电压匹配的高压锂电 / 直流电源                      |
| 急停        | 物理 E-Stop 按钮，串接在电机母线上（**绝对必备，CAN 关断不可替代**） |

**接线约定**：左腿电机（CAN ID 1–6）走 `can_usb`，右腿电机（CAN ID 7–12）走 `can_spi`。两条总线可在 `run.py` 的 `--can-usb` / `--can-spi` 参数中重命名。

---

## 3. 软件环境准备

### 3.1 系统依赖（Ubuntu / 树莓派 OS）

```bash
sudo apt update
sudo apt install -y \
    git build-essential \
    can-utils iproute2 \
    libsdl2-2.0-0 libsdl2-dev \
    bluetooth bluez \
    python3-pip
```

### 3.2 Python 3.12 与 `uv`

本项目要求 **Python ≥ 3.12**（见 `.python-version` 与 `pyproject.toml`）。推荐使用 [`uv`](https://github.com/astral-sh/uv) 管理虚拟环境与依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                # 让新安装的 uv 进入 PATH
uv --version
```

如果使用系统 pip，请先 `pyenv` 或 `apt` 安装 Python 3.12，并自行创建虚拟环境。

---

## 4. 拉取与依赖安装

```bash
git clone <仓库地址> python_olaf_sdk
cd python_olaf_sdk

# 一行同步：创建 .venv，按 uv.lock 安装全部依赖
uv sync

# 进入虚拟环境（可选；下文也可用 `uv run` 替代）
source .venv/bin/activate
```

依赖列表（节选自 `pyproject.toml`）：

```text
numpy           # 观测拼装、矩阵运算
python-can      # SocketCAN
onnxruntime     # 策略推理
pyyaml          # 配置解析
damiao-motor    # 达妙电机库（如使用）
pygame          # 手柄
pyserial        # IMU 串口
tqdm, colorama  # 工具
```

### 4.1 放置策略模型

把训练侧导出的 ONNX 文件命名为 `policy.onnx`，放入仓库根目录。`policy.py` 默认读取 `./policy.onnx`。

> 提示：`.gitignore` 把 `*.onnx` 排除在外，Git 不会上传策略本体。

---

## 5. 硬件接口配置

### 5.1 CAN 接口命名（重要）

`run.py` 默认期待两条总线名 `can_usb`（左腿）与 `can_spi`（右腿）。可以在系统层用 udev 给设备起别名，也可以在启动时用 CLI 覆盖：

**临时启用（开机后每次手动）：**

```bash
sudo ip link set can_usb up type can bitrate 1000000
sudo ifconfig can_usb txqueuelen 1000

sudo ip link set can_spi up type can bitrate 1000000
sudo ifconfig can_spi txqueuelen 1000
```

**或在 `run.py` 启动时指定别名：**

```bash
uv run python run.py --can-usb can0 --can-spi can1
```

**只用一条总线先调试：**

```bash
uv run python run.py --bus usb         # 只驱左腿；右腿保持 DEFAULT_JOINT_POS
uv run python run.py --bus spi         # 只驱右腿
```

### 5.2 IMU 串口

`imu.py` 默认通过 by-id 路径打开 CP2102：

```text
/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0
```

接好串口后用 `ls /dev/serial/by-id/` 确认设备出现。波特率 921600。如果使用其他芯片（如 CH340），请在 `imu.py:DEFAULT_PORT` 中改为对应路径或 `/dev/ttyUSB0`。

### 5.3 手柄（Xbox One S）

```bash
# 驱动加载
sudo modprobe xpad joydev
echo -e "xpad\njoydev" | sudo tee /etc/modules-load.d/xbox.conf

# 用户加入 input 组（注销重登或 newgrp input 生效）
sudo usermod -aG input $USER
```

**蓝牙配对：**

```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# scan on
# 控制器上同时按 Xbox 键 + Pair 键，等手柄 MAC 出现
[bluetooth]# pair  <MAC>
[bluetooth]# trust <MAC>
[bluetooth]# connect <MAC>
```

确认接入：`ls /dev/input/js*` 应当看到 `js0`。

> **提示**：手柄是**必需**的——`A` 键是急停，开机时若未检测到手柄，`run.py` 会直接报错退出。

---

## 6. 关键参数核对（**上电前必做**）

策略对训练时的物理量约定极度敏感，**任何一项不一致都会让机器人瞬间摔倒**。下面这五项必须逐项核对：

1. **关节顺序** —— `config.JOINT_ORDER` 必须与训练环境的 `LEG_JOINT_NAMES` 完全一致（`[L, R]` 交错）。一旦顺序错乱，策略输出像"差不多在工作"但很快发散。
2. **CAN ID** —— 在 `config.MOTOR_TABLE` 中按实物接线填好每个关节的 `can_id`。
3. **方向标志** —— `MotorSpec.direction = ±1`，使关节正向旋转方向与 URDF 一致。出厂值是按旧的非对称 `DEFAULT_JOINT_POS` 标定的，**右腿如膝/髋俯仰需在调试时复核**。
4. **IMU 安装方向** —— 观测使用 root 帧的投影重力与角速度。`imu.py:R_base_from_imu` 默认是单位阵；如果芯片换装方向，必须修改此矩阵（仅允许偏航 90° 倍数旋转，**不要改 z 行**）。
5. **观测拼接顺序与归一化** —— `observation.py` 与 `policy.py` 的实现必须与训练侧逐字段对应。RunningStandardScaler 的 mean / var 已写入 ONNX 元数据，由 `policy.py` 自动读取。

> 推荐做法：在训练机用 `play.py` 打印一组 sim obs，在 Pi 上用 `--joints l_hip_yaw` 单关节模式打印同姿态的 obs，逐字段 diff。

---

## 7. 首次上电流程

> 操作前确认：物理急停按钮在手边、机器人**已悬挂或被人扶住**、关节工作空间内无人。

### 7.1 启动顺序

```bash
# 1. 启用两条 CAN 总线
sudo ip link set can_usb up type can bitrate 1000000
sudo ifconfig can_usb txqueuelen 1000
sudo ip link set can_spi up type can bitrate 1000000
sudo ifconfig can_spi txqueuelen 1000

# 2. 检查 IMU、手柄
ls /dev/serial/by-id/   # 应看到 CP2102
ls /dev/input/js*       # 应看到 js0

# 3. 启动主程序（建议第一次加 --slomo 慢速模式）
uv run python run.py --slomo
```

启动后控制台会出现：

```text
joystick attached — left stick drives velocity_cmd (max 0.60 m/s)
IMU opened on /dev/serial/by-id/... @ 921600 baud
mode → IDLE
Ready. Buttons: X → zero pose, Y → DEFAULT_JOINT_POS, B → ONNX policy, RB → encoder-zero calibration, A → EMERGENCY STOP.
q: L_HIP_YAW [+0.000] | ...
```

此时电机已使能但 `kp = 0`，腿处于"软挂"状态——可以用手轻松扳动。

### 7.2 走起来的动作链

| 步骤 | 操作            | 状态机变化                       | 物理表现                                 |
|------|-----------------|----------------------------------|------------------------------------------|
| 1    | 启动 `run.py`   | `IDLE`                           | 电机使能、kp=0，腿可手动拖动             |
| 2    | 按 **Y**        | `MOVE_DEFAULT`                   | 同步斜坡到 `DEFAULT_JOINT_POS`（蹲姿） |
| 3    | 等姿态稳定      | 仍在 `MOVE_DEFAULT`              | 控制台心跳显示 `q ≈ 默认姿态`            |
| 4    | 按 **B**        | `POLICY`                         | 策略接管；左摇杆给 `velocity_cmd`        |
| 5    | 摇杆推杆        | —                                | 机器人踏步前进/后退/侧移                 |
| 6    | 按 **A**        | 急停 latched                     | 立即阻尼，进程退出                       |

**B 键有保护**：当前测得姿态与 `DEFAULT_JOINT_POS` 的 L∞ 误差必须小于 `--pose-tolerance`（默认 0.15 rad），否则 B 会被忽略并提示先按 Y。

---

## 8. 关节零点标定

电机出厂或更换后必须做一次零点标定，否则关节读数与 URDF 模型对不上。

**操作流程：**

1. `run.py` 启动后处于 `IDLE`。
2. 按 **RB** —— 进入 `CALIBRATE` 模式：电机变软（kp=0），控制台以 5 Hz 打印 12 个关节当前角度。
3. 用手把每一个关节摆到你希望定义为"0 rad"的姿态。
4. 再次按 **RB** —— SDK 会：
   - 暂停电机线程；
   - 向每个 Robstride 电机发送 `SET_ZERO_POSITION (0x06)` + `SAVE_PARAMETERS (0x16)`；
   - 写入 `calibration.json` 作为审计日志（**仅用于诊断，不会被 SDK 重新加载**）；
   - 回到 `IDLE`。

零点已写入电机 flash，**断电不丢失**。下次开机无需重做。

---

## 9. 手柄使用与按键说明

| 按键 | 模式跳转                | 说明                                                       |
|------|--------------------------|------------------------------------------------------------|
| **Y**  | → `MOVE_DEFAULT`         | 同步斜坡到 `DEFAULT_JOINT_POS`（蹲姿）                     |
| **X**  | → `MOVE_ZERO`            | 同步斜坡到全零姿态（URDF zero）                            |
| **B**  | → `POLICY`               | 交给策略；当前姿态需在容差内（先按 Y）                     |
| **A**  | → 急停 latched           | 立即阻尼并退出，**不可恢复**，需重启程序                   |
| **RB** | IDLE ⇄ CALIBRATE         | 进入/写入零点标定                                          |
| **左摇杆 X** | —                  | `+vy`（左）/ `-vy`（右）                                   |
| **左摇杆 Y** | —                  | `+vx`（前）/ `-vx`（后）                                   |
| **右摇杆 X** | —                  | `+wz`（左转）/ `-wz`（右转）                               |

速度命令的硬限幅与训练分布对齐：`vx ∈ [-0.4, 0.7]`、`vy ∈ [±0.4]`、`wz ∈ [±1.0]`。摇杆死区 0.08，超出死区后线性重映射至全量程。

### 9.1 手柄独立验证

```bash
uv run python joystick_dump.py            # 打印每个按键 / 轴的索引与值
uv run python joystick.py --max-vel 1.0   # 实时显示 vx/vy/wz
```

不同驱动（xpad / xone / bluez）下右摇杆 X 轴的索引可能从 3 变成 2，请以 `joystick_dump.py` 打印的索引为准，必要时修改 `joystick.py:_AXIS_RIGHT_X`。

---

## 10. 调试与单关节验证

| 标志                   | 作用                                                                           |
|------------------------|--------------------------------------------------------------------------------|
| `--slomo`              | 策略下发的目标位置 slew 限幅到 `SLOMO_VMAX_RAD_S`（默认 0.5 rad/s）            |
| `--joints L,R,…`       | 仅指定关节跟踪策略；其余保持 `DEFAULT_JOINT_POS`，机器人继续撑住                |
| `--debug`              | 按 B 后只跑 `--debug-actions` 步策略，每步 \|Δq\| ≤ `--debug-dq`，然后正常退出 |
| `--debug-actions N`    | `--debug` 模式的步数（默认 10）                                                |
| `--debug-dq RAD`       | `--debug` 模式的单步关节增量上限（默认 0.1 rad）                               |
| `--bus usb / spi / both` | 限定下发到某条总线（不影响策略推理维度）                                     |
| `--log-level DEBUG`    | 打印每帧 SocketCAN trace（极其冗长，仅作排查用）                               |
| `--pose-tolerance R`   | B 键放行的 `DEFAULT_JOINT_POS` 容差（默认 0.15 rad）                           |

### 推荐调试节奏

```bash
# 第一次上电：单关节 + 慢速 + 调试预算
uv run python run.py --slomo --debug --joints l_knee_pitch

# 单腿踝部三关节，慢速
uv run python run.py --slomo --joints l_ankle_pitch,l_ankle_roll,r_ankle_pitch,r_ankle_roll

# 全部关节，慢速
uv run python run.py --slomo

# 全速（确认 slomo 也稳了再放开）
uv run python run.py
```

### IMU 单独验证

```bash
uv run python imu_dump.py --rate 10
```

四组倾斜测试（前/侧/静止/偏航）观察 `proj_g` 与 `ang_vel`：静止时 `proj_g ≈ (0, 0, -1)`；前倾使 `proj_g_x` 变正。

---

## 11. 安全机制

SDK 内置多层安全网，但**机械急停永远是最后一道防线**：

- **物理 E-Stop**：必须串接在电机供电母线上。CAN 总线断开不可代替。
- **kp 软斜坡**：模式切换后 2 秒内 kp 从 0 线性升到 1，避免硬启停。
- **看门狗**：策略 tick > 40 ms 时电机线程切换为 `damp_all()`（kp=0、kd>0）。
- **NaN/Inf 守护**：
  - IMU 四元数零模或非有限值 → 退化为 `(0, 0, -1)`，角速度归零并打印 warning；
  - 电机线程收到非有限目标 → LPF 重置 + 当帧阻尼。
- **软关节限位**：策略输出经 `JOINT_LIMITS_SOFT`（URDF 范围的 95%）裁剪，与训练侧 `soft_joint_pos_limit_factor=0.95` 对齐。
- **力矩软上限**：`MotorSpec.tau_limit` 按训练 `effort_limit_sim` 限制估计输出力矩（RS02 11.9 N·m / RS03 42 N·m / RS00 14 N·m）。
- **A 键急停**：latched，在任意线程触发立即 `damp_all()` 并 `_stop.set()`，无法在线复位，必须重启进程。

---

## 12. 常见问题排查

| 现象                                                    | 可能原因 / 处理                                                              |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| `RuntimeError: No joystick detected`                    | 未接手柄 / 蓝牙未连 / 当前用户不在 `input` 组                                |
| 启动报 `[Errno 19] No such device` (CAN)               | `can_usb` / `can_spi` 没 `ip link set up`，或别名不对                        |
| `non-finite IMU angular velocity` warning               | 串口数据帧错位；检查波特率、线材，重启 IMU                                   |
| `non-finite target, damping this tick`                  | 策略输出 NaN，多半是观测里某项异常（IMU/编码器）；先用 `--joints` 单关节验证 |
| 右腿走的方向反了                                        | 复核 `MOTOR_TABLE` 中右腿 `direction`；用 `--slomo --joints` 单关节确认      |
| 按 B 没反应、提示 `B ignored — max joint error ...`    | 当前姿态远离 `DEFAULT_JOINT_POS`，先按 Y 等姿态稳定；或加 `--pose-tolerance 0.25` |
| `policy tick X.X ms > watchdog`                         | Pi 算力不够或被其他进程抢占；`taskset` 绑核、降低日志等级                    |
| 启动后 12 个关节 `q` 全是 0，且 IMU 也是 0              | 电机/IMU 通讯失败；用 `candump can_usb` 看是否有反馈帧                       |
| 倾斜机器人时 `proj_g` 变化方向不对                      | `imu.py:R_base_from_imu` 安装矩阵不对，按 `imu.py` 注释修改                  |
| 摇杆推前但机器人后退                                    | 轴反相，参考 `joystick.py:VX_RANGE` / `_AXIS_LEFT_Y`                         |

---

## 13. 文件结构速查

```text
python_olaf_sdk/
├── run.py              # 主入口（状态机、策略循环、电机循环）
├── config.py           # 关节顺序、默认姿态、CAN ID、限位、回路频率
├── motors.py           # Robstride / 达妙 MIT 模式驱动
├── can_bus.py          # SocketCAN 薄封装
├── imu.py              # HI226/HI229 串口解析（独立线程）
├── observation.py      # 观测拼装 + path-frame
├── policy.py           # ONNX 推理 + RunningStandardScaler
├── joystick.py         # Xbox 手柄 → velocity_cmd
├── imu_dump.py         # IMU 调试工具
├── joystick_dump.py    # 手柄按键/轴索引调试工具
├── policy.onnx         # 训练侧导出的策略（不入 Git）
├── pyproject.toml      # 依赖声明
├── uv.lock             # 锁定版本
└── README.md           # 本手册
```

---

## 速查命令清单

```bash
# 环境
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && source .venv/bin/activate

# CAN 启用
sudo ip link set can_usb up type can bitrate 1000000 && sudo ifconfig can_usb txqueuelen 1000
sudo ip link set can_spi up type can bitrate 1000000 && sudo ifconfig can_spi txqueuelen 1000

# 工具
uv run python joystick_dump.py
uv run python imu_dump.py --rate 10

# 上电（建议组合）
uv run python run.py --slomo --debug --joints l_knee_pitch    # 第一次单关节
uv run python run.py --slomo                                  # 全关节慢速
uv run python run.py                                          # 全速
```

> 任何时候，按 **A** 急停，掉电由物理 E-Stop 兜底。