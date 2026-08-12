# RoboJuDo DDS Sim/Real 统一迁移计划

> **状态：Phase 1 已完成；Phase 2 及以后待确认。** 本文不表示当前 `g1` 已使用 DDS，也不授权运行
> `g1_real`、连接机器人或删除现有 MuJoCo 路径。
>
> **推荐决策：** 先保留当前 `g1`，通过一个不注册为公开配置的 DDS 仿真 harness 验证
> `domain 1 + lo`。外部 `unitree_mujoco` 只作为阶段性的协议参考和对照测试，不作为最终运行依赖。
> 理想终态是在 RoboJuDo 内维护最小的 G1 DDS simulator；公开路径切换和旧直连 adapter 的删除仍分别需要
> Gate A、Gate C 后的再次确认。

## 1. 目标与非目标

本计划希望让策略侧通过同一个 Unitree HG DDS 合同连接仿真和实机：

```text
g1      -> Unitree DDS (domain 1, lo)  -> RoboJuDo G1MujocoDdsServer
g1_real -> Unitree DDS (domain 0, NIC) -> physical G1
```

目标：

- 保持唯一的 G1 29DoF、WoGait checkpoint、480→29 观测/动作合同和 50 Hz policy loop。
- 将 endpoint 表示为不可拆散的 deployment profile，而不是任意 `domain_id`/`net_if` 组合。
- 先证明 DDS state/command、关节顺序、CRC、tick、IMU、时序和故障行为，再讨论取消旧直连 adapter；
  MuJoCo plant 和仓内资产继续由第一方 server 复用。
- 保持 `python scripts/run_pipeline.py -c g1|g1_real` 的名称；切换后 `g1` 的操作流程会变成两个进程。
- 每一阶段可单独测试、提交和回滚，不把配置清理、通信切换、物理模型切换混为一个 patch。
- 复用 RoboJuDo 当前 G1 29DoF XML、物理参数、力矩限制和 MuJoCo 依赖；最终运行不要求 sibling
  `unitree_mujoco` checkout。
- 只实现 RoboJuDo 客户端实际消费的最小 HG DDS 合同，不复制官方 simulator 的 UI、BMS、secondary IMU、
  SportMode 等未使用功能。

非目标：

- 不增加其他机器人、低 DoF 策略、motion/tracking 或训练代码。
- 不把 DDS domain 当作权限、安全认证或硬件急停。
- 不自动探测/切换网卡，不在失败时从 Domain 1 fallback 到 Domain 0。
- 不在同一进程同时初始化多个 DDS endpoint；Unitree `ChannelFactory` 是进程级单例。
- 不在本迁移的第一阶段引入 Hydra、multirun、远程 launcher 或 `_target_` 动态实例化。
- 不宣称官方 `unitree_mujoco` 与当前 MJCF 具有逐点相同的动力学。
- 不直接复制整个 `unitree_mujoco` 源码或把它变成 submodule/vendor；其代码和固定 commit 只用于行为对照。
- 不把 DDS 发布/订阅、物理步进和 viewer 再塞进一个新的“大而全”环境类。

## 2. 当前事实与候选架构

| 路径 | 当前环境 | 进程边界 | DDS | 输入 | 生命周期 |
| --- | --- | --- | --- | --- | --- |
| `g1` | 进程内 `MujocoEnv` | 单进程同步 `mj_step` | 无 | 本地 pygame 手柄 | 不执行 ramp/blend |
| `g1_real` | `UnitreeCppEnv` | Python/C++ + 机器人 | 显式 hardware / Domain 0 / 实机网卡 | G1 remote bytes | 3 秒 ramp + 5 秒 blend |

当前关键限制：

- Domain 已贯通配置与 C++ binding，但公开 `g1` 仍是进程内 MuJoCo；`simulation/1/lo` 目前只是未注册的配置合同，
  还没有 DDS simulator/client 运行路径。
- C++ writer 每 20 ms 重发最后一个目标；command 和 state 都没有 freshness watchdog。
- `self_check()` 只检查缓存中 `tick != 0`，CRC 错包虽被丢弃，但旧缓存仍可无限继续使用。
- MotionSwitcher release 在构造期执行且没有总 deadline；构造器同时创建 publisher、subscriber 和 writer。
- `joint2motor_idx` 若非空只重排反馈，发送目标时没有逆映射；迁移期间必须保持 identity。
- `is_sim` 同时影响 prepare、掉帧处理等生命周期，不能把它等同于“是否使用 DDS”。

推荐的过渡架构：

```text
PolicyLoop
├── InProcessMujocoPort                # 公开 g1，Gate C 前保留
└── UnitreeDdsPort
    ├── simulation / domain 1 / lo     # 第一方 server + 临时 client harness
    └── hardware / domain 0 / real NIC # 公开 g1_real
```

Gate C 通过后的候选终态：

```text
                         +-> domain 1 / lo  -> RoboJuDo G1MujocoDdsServer
PolicyLoop -> UnitreeDdsPort
                         +-> domain 0 / NIC -> physical G1
```

该终态让 policy/client 的 sim/real I/O 完全一致，并消除外部 simulator 运行依赖；代价是 RoboJuDo 必须维护一个
独立的 DDS simulator 进程。为控制复杂度，MuJoCo plant、直连 adapter、DDS server 和 viewer 必须组合而不是互相继承，
且 DDS server 只覆盖本仓使用的 G1 29DoF LowCmd/LowState 合同。

## 3. 固定数据合同

### 3.1 Deployment profile

建议先定义数据，再改装配逻辑：

```text
DdsEndpoint
  domain_id: int
  interface: str

DeploymentProfile
  target: simulation | hardware
  endpoint: DdsEndpoint
  input_source: local_joystick | unitree_remote
  prepare_enabled: bool
  safety_check: bool
```

只允许以下原子组合：

| Profile | Target | Domain | Interface | Input | Prepare | Safety |
| --- | --- | ---: | --- | --- | ---: | ---: |
| simulation | first-party MuJoCo DDS server | 1 | 固定 `lo` | 第一阶段保留 local joystick | false | false |
| hardware | real G1 | 0 | 启动时必填、存在且非 loopback | Unitree remote | true | true |

约束：

- Domain 0/1 是 Unitree 采用的发现隔离约定，不具有固有的 sim/real 含义。
- `eth0` 只是示例，不是默认值；真实网卡名不得提交到共享 preset。
- `simulation + 非 lo`、`hardware + lo`、空网卡、未知 target 必须在初始化 DDS 前失败。
- Domain 和 target 不提供自由 CLI override；真实网卡建议由显式 `--net-if <name>` 提供。
- 不根据收到什么 topic、是否安装 SDK 或网络是否可达来猜测 target。
- 一个进程第一次初始化 endpoint 后，再以不同 domain/interface 构造 controller 必须报错。

若以后采用 Hydra，它只可选择整个 `deployment=simulation|hardware` profile；不得单独 sweep domain、网卡、
safety、checkpoint、关节顺序、频率或 gains。

### 3.2 DDS wire contract

| 项目 | 固定值/规则 |
| --- | --- |
| IDL | Unitree HG |
| State | `rt/lowstate`, `LowState_` |
| Command | `rt/lowcmd`, `LowCmd_` |
| Motors | G1 前 29 个 motor slots，identity motor order |
| State | `q[29]`, `dq[29]`, `tau_est[29]`, IMU wire quaternion `[w,x,y,z]`/gyro, remote bytes, tick, mode_machine |
| Command | finite `q_target[29]`, `dq=0`, `tau=0`, policy Kp/Kd，最后计算 CRC |
| Policy rate | 50 Hz / 20 ms |
| Simulator bridge | 1 ms state/PD worker；policy target 在 20 ms 周期更新 |

接收顺序必须是：CRC → tick → finite-value → mode → 原子提交 state 和 freshness。失败包不得刷新时间戳。
Tick 的 duplicate、forward、wrap 和 regression 需要无符号模运算测试，所有 age 使用 monotonic clock。

### 3.3 DDS lifecycle

如果 DDS 路径成为默认 backend，C++ 侧必须显式区分以下阶段，而不能靠“对象构造成功”代表可发命令：

| Phase | 进入条件 | LowCmd 行为 |
| --- | --- | --- |
| `BOOTSTRAP` | 初始化 endpoint/observer | 不发布策略目标 |
| `READY` | 至少两个 CRC 正确、tick 前进、mode 稳定的新鲜 state | 仍不发布策略目标 |
| `ACTIVE` | operator gate 通过并原子 `arm(initial_target)` | 发布策略目标 |
| `FAULT` | state/command stale、tick 回退、mode 改变、非法 target 或 stop | 执行经批准的 failure output，fault 锁存 |
| `STOPPED` | writer 已停止并完成 teardown | 拒绝新目标 |

failure output 目前尚未最终决定。现有 `Kp=0, Kd=5` damping 不是断使能；正式 timeout 和 fault 动作必须先在
loopback/仿真中测量，再经硬件负责人确认。禁止留下可由普通 CLI 关闭的 watchdog 开关。

## 4. 第一方 DDS simulator 与外部参考边界

### 4.1 为什么不把外部 simulator 作为终态

当前 `MujocoEnv` 已经具备 29DoF 模型加载、1 ms 物理子步、每子步 PD 计算、torque clip、状态读取、reset 和
viewer。RoboJuDo 缺少的是 HG DDS server 边界，而不是另一套物理引擎。长期依赖外部 `unitree_mujoco` 会引入第二份
MJCF/mesh、不同物理参数、独立构建和版本锁定，并让 RoboJuDo 的仿真结果受 sibling checkout 影响。

因此，外部项目只用于回答“官方 topic/type/字段/时序大致如何表现”；最终实现和自动测试必须仅依赖 RoboJuDo
checkout、已 vendored 的 Unitree C++ binding/SDK2，以及当前 MuJoCo 资产。

### 4.2 第一方组件边界

不要直接给当前 `MujocoEnv` 增加 DDS callback。先提取无通信、无策略依赖的物理核心，再用两个薄 adapter 组合：

```text
G1MujocoPlant
├── DirectMujocoEnv                 # 迁移期保留；同步策略直连
└── G1MujocoDdsServer               # 独立进程；domain 1 / lo
    ├── HG LowCmd subscriber
    ├── 1 ms PD/physics loop
    └── HG LowState publisher
```

`G1MujocoPlant` 负责模型/data、reset、读取 q/dq/base/IMU、施加 torque、`mj_step` 和可选 viewer；它不知道
DDS、policy、controller 或 deployment profile。`DirectMujocoEnv` 保持当前 `pd_target[29]` 行为，作为迁移基线和
快速确定性测试。`G1MujocoDdsServer` 是独立进程，不能与 policy client 共享对象或绕过 DDS。

第一方 server 的最小功能面：

- 订阅 HG `rt/lowcmd`，缓存完整的 `q/dq/tau/kp/kd/mode_machine/CRC`，拒绝 CRC 错误、非法数值和错误 mode。
- 每个物理子步用最新有效命令计算 `tau + kp*(q_cmd-q) + kd*(dq_cmd-dq)`，按当前 RoboJuDo torque limits
  限幅后推进 MuJoCo；进入 active 后命令超时会触发锁存 fault，不无限保持旧 target。
- 发布 HG `rt/lowstate`，至少填充 29 个 q/dq/tau_est（使用实际裁剪后 actuator torque）、IMU wire quaternion
  `[w,x,y,z]`/gyro、递增 tick、`mode_machine=5`、默认全零的 `wireless_remote[40]` 和最后计算的 CRC；client
  继续在唯一边界转换为内部 `[x,y,z,w]`。
- 明确 readiness、command freshness、shutdown 和 simulator-exit 行为；server fault 后停止 active command，
  通过非零进程退出和 LowState stale 让 client/supervisor 发现，不能伪造非标准 DDS fault 字段或静默 fallback。
- 不实现当前 RoboJuDo 未消费的 BMS、secondary IMU、SportModeState 或通用多机器人分支；若以后确有消费者，
  以单独合同和测试加入。

当前 `MujocoEnv.step()` 只接收 position target，并使用环境内固定 Kp/Kd；它不能原样充当 LowCmd server。
提取 plant 后，直连 adapter 继续生成现有控制行为，DDS adapter 则完整解释 LowCmd。两者必须用 golden test 证明
在 `dq=0/tau=0` 且 gains 相同时产生逐元素相同的 torque。

server 生命周期与 client 的 `BOOTSTRAP/READY/ACTIVE/FAULT` 门禁分开定义：

| Server phase | 条件 | 行为 |
| --- | --- | --- |
| `OBSERVE` | DDS/plant 已启动，尚无有效 LowCmd | 发布 LowState；不启用 command timeout，不施加 active target |
| `ACTIVE` | 收到首个 CRC/mode/finite 均有效的 LowCmd | 每子步执行 PD；只接受更新且开始 command freshness 计时 |
| `FAULT` | ACTIVE 后 command stale、非法新命令、physics 非有限或内部异常 | 拒绝后续 active command，执行有界仿真 failure action，然后非零退出/停止 LowState |
| `STOPPED` | 正常 teardown 完成 | 停止 DDS 和 physics；拒绝重启同一对象 |

因此 Phase 3 的 state-only 验证可以长期停留在 `OBSERVE`，不会因没有 command publisher 自行 fault；command
watchdog 只在进入 `ACTIVE` 后生效。server 不自行解释 operator arm 权限，真实的 arm/readiness 门禁仍由 client C++
controller 承担。

DDS server 侧优先扩展 vendored `unitree_cpp` 的 C++ binding，暴露 server-role publisher/subscriber 和纯数据快照；
Python 只负责 MuJoCo plant loop。这样不增加 `unitree_sdk2py` 依赖，也不把 SDK 消息/CRC 细节复制到 Python。
若 1 ms wall-clock 测试证明 Python orchestration 不稳定，再把 physics/bridge loop 下沉到一个小型 C++ 可执行文件；
不能在没有测量前复制官方完整 simulator。

### 4.3 外部参考实现

已克隆并审计：

- `../unitree_mujoco`：`ae6a8403e272733e9996ef59990880330496177f`
- `../G1-Playground`：`90ec9961a369ec9f1e8e034116cf9b31c1a7bb6e`

对照测试使用官方 C++ simulator，而不是 Python bridge。Python bridge 当前没有完整建立 tick/CRC 合同；不能为了让
仿真通过而放松真实机检查。C++ 模拟器的参考启动参数为：

```bash
cd ../unitree_mujoco
cmake -S simulate -B simulate/build
cmake --build simulate/build --parallel
./simulate/build/unitree_mujoco \
  -r g1 -s scene_29dof.xml -i 1 -n lo
```

RoboJuDo 不自动 clone、build、import 或启动该 sibling checkout。它只在阶段性 L3 对照中由人工显式启动；普通安装、
CI、`g1` 和最终 DDS simulator 都不得检查该目录是否存在。迁移完成后仍不使用 Git submodule/vendor，文档只记录
作为对照基准的 commit、SDK2、CycloneDDS、MuJoCo 和编译器版本。

两个模型均为 `nq=36/nv=35/nu=29`，29 个 actuator/joint 顺序和初始 qpos 相同，但它们不是同一物理模型：

- 当前模型总质量约 `33.341142 kg`，官方模型约 `35.112142 kg`。
- 当前 runtime 强制 `0.001 s` timestep，官方模型默认 `0.002 s`。
- damping、friction loss、ctrl range、腰部/躯干惯量和 sensor layout 均存在差异。
- 官方 bridge 假定前 87 个 sensordata 是 29 position + 29 velocity + 29 torque；不能直接加载当前仅少量
  sensor 的 XML。

因此可比较 topic/type、字段、CRC、tick、motor order 和基本时序；闭环轨迹只能做行为验收，不能要求逐点相等。
第一方 server 使用 RoboJuDo 当前模型作为权威物理模型，不为了匹配外部模型而改动质量、damping 或传感器布局。

## 5. 分阶段实施计划

### Phase 0 — 冻结基线

1. **隔离当前未提交文档工作并创建 `pre-dds` 基线** → verify：工作树没有与 DDS 无关的 staged/unstaged 混合，
   checkpoint、vendored revisions 和 `unitree_mujoco` commit 均有记录。
2. **保存 `g1/g1_real` resolved 配置与有效控制数据** → verify：记录 env/policy joint orders、adapter indices、
   default pose、Kp/Kd、limits、50 Hz、480→29 shape 和 Unitree binding dict。
3. **运行现有静态门禁与人工进程内 MuJoCo 基线** → verify：unittest、Ruff、format、diff-check 通过；人工记录
   零命令、x/y/yaw、小扰动、跌倒和 shutdown 行为。

执行证据记录在 [`docs/pre_dds_baseline.md`](docs/pre_dds_baseline.md)，机器可读合同与回归测试分别位于
[`tests/fixtures/pre_dds/contract.json`](tests/fixtures/pre_dds/contract.json) 和
[`tests/test_pre_dds_contract.py`](tests/test_pre_dds_contract.py)。手柄与实机行为仍属于人工验收，不能以自动测试
替代。

> [!NOTE]
> Phase 0 已于 2026-08-11 过门：自动合同、数值 golden、静态检查、零输入仿真启动和 `pre-dds` 回滚标签均已
> 完成；用户确认手柄基线在 sim 和 real 均正常。顶层 `third_party/README.md` 为用户有意删除，过时的存在性断言
> 已同步移除。实机结果属于 operator-reported evidence，Codex 没有执行硬件动作。

### Phase 1 — 只贯通 Domain，不改变公开路径

1. **保护 pre-DDS 证据不可变** → verify：不得修改 `tests/fixtures/pre_dds/contract.json` 中的旧 vendor hash；C++
   变更触发旧树哈希失败后，将“当前树必须等于旧树”的断言迁移为 Phase 1 focused contract，只允许下述 Domain
   plumbing 文件发生有解释的变化，并继续执行其余 pre-DDS 数值/配置合同。
2. **为 Unitree config 加入 `domain_id`** → verify：值依次经过 Pydantic → dict → pybind → C++
   `UnitreeConfig` → `ChannelFactory::Init(domain_id, net_if)`，RoboJuDo 自身始终显式传值。
3. **保持 `g1_real` 行为兼容** → verify：它仍固定 Domain 0；除新增字段和 endpoint 日志外，topics、gains、频率、
   prepare 和 safety 均与基线一致。
4. **增加 endpoint 交叉校验** → verify：`simulation/1/lo` 和 `hardware/0/<NIC>` 通过；所有交叉组合在 C++
   controller 构造前失败。
5. **保护 ChannelFactory singleton** → verify：同进程第二次使用不同 endpoint 明确失败，而不是静默复用第一次配置。
6. **保持 optional import** → verify：没有 UnitreeCpp/SDK2 的机器仍可导入并运行当前 `g1`。

> [!NOTE]
> Phase 1 Step 1 已于 2026-08-12 完成。pre-DDS 合同保持原始 SHA256
> `d43db515848107099105f3ae2e891097410b7810bfd4ecf14d3a5ba0dbeb240b`；新的
> [`unitree_cpp_boundary.json`](tests/fixtures/dds_phase1/unitree_cpp_boundary.json) 与
> [`test_dds_phase1_boundary.py`](tests/test_dds_phase1_boundary.py) 将后续允许变化的范围精确限制为
> `example/config.py`、`src/py_binding.cpp`、`src/unitree_controller.cpp` 和
> `src/unitree_controller.hpp`。新增、删除或修改其他 vendored 文件都会触发失败。此步骤仅建立证据边界，尚未修改
> Domain、C++ 或运行时行为。

> [!NOTE]
> Phase 1 Step 2 已于 2026-08-12 完成。`domain_id` 现在是 RoboJuDo `UnitreeCfg` 的必填字段，`g1_real`
> 在组合点显式传入 `0`；`model_dump()` 将其原样交给 pybind，后者强制读取该键并写入
> `UnitreeConfig::domain_id`，最终调用 `ChannelFactory::Init(cfg_.domain_id, cfg_.net_if)`。聚焦测试位于
> [`test_dds_phase1_domain.py`](tests/test_dds_phase1_domain.py)。本机已在 `/tmp` 使用实际 SDK2/CycloneDDS
> 编译扩展并验证 native 字段可读写，缺少该键会在创建控制器和 DDS participant 前抛出 `KeyError`。本步骤没有
> 实例化真实控制器、连接机器人或实现后续 endpoint 组合校验与 singleton 保护。

> [!NOTE]
> Phase 1 Steps 3–6 已于 2026-08-12 完成。`g1_real` 的旧 composition、topics、50 Hz、gains、prepare/safety
> 合同继续由 pre-DDS fixture 验证，endpoint 固定为 `hardware/0/<non-lo>`。未注册的 simulation profile 只接受
> `simulation/1/lo`；交叉 Domain/interface、空白值及非严格整数都会在 Pydantic 和环境副作用边界失败。这里的
> `<non-lo>` 是结构检查，不探测当前机器上接口是否存在或是否带有内核 loopback flag，真实 NIC 仍须按硬件
> preflight 验证。
>
> C++ `DdsEndpointInitGuard` 只在首次 SDK 初始化成功后记录 endpoint；相同 endpoint 复用、不同 endpoint 抛错，
> 初始化异常可重试，并发同 endpoint 只调用 initializer 一次。纯 C++ 测试使用 fake initializer，不创建 DDS；完整
> extension 已在 `/tmp` 对实际 SDK2/CycloneDDS 编译通过，但未构造真实 `UnitreeController`。独立进程测试还证明
> 强制屏蔽 `unitree_cpp` 时，`g1` 的配置、MuJoCo lazy import 和 480→29 推理保持可用。该 guard 只保护经本
> extension 发起的初始化；SDK 没有 endpoint 查询接口，无法识别同进程其他代码此前直接调用的
> `ChannelFactory::Init`。

该阶段不得顺手删除 Registry、重命名环境类或调整 pipeline 时序；这些会污染 domain plumbing 的回归定位。

### Phase 2 — 建立只读 readiness 与可观测性

1. **拆开 observer 与 command activation** → verify：state-only 模式只创建 subscriber，不释放 motion service、不创建
   command writer，也不会因为 `act=False` 仍产生 LowCmd。
2. **实现 coherent state status** → verify：CRC 错误、NaN/Inf、重复 tick 和 tick 回退不会更新 freshness；合法 tick
   wrap 被接受；mode 与 state 在同一同步边界提交，不存在 callback/writer data race。
3. **暴露只读 status** → verify：至少包含 phase、fault、endpoint、tick、mode、state age、command age 和错误计数；
   状态转换只记录一次日志，不在实时 callback 刷屏。
4. **限制 MotionSwitcher 副作用** → verify：simulation/read-only 完全跳过 release；hardware release 有总 deadline，失败时
   不创建 command publisher。

### Phase 3 — 提取 MuJoCo plant 并建立第一方 state-only server

1. **冻结当前直连 torque/state golden** → verify：保存固定 q/dq/target/gains 下每个物理子步的 torque、下一状态、
   quaternion 顺序和 reset 结果，后续 refactor 逐元素一致。
2. **提取 `G1MujocoPlant`** → verify：模型、data、1 ms step、torque clip、state、reset 和 viewer 从 policy/environment
   装配中解耦；公开 `g1` 仍经薄 `DirectMujocoEnv` 运行且 golden 不变。
3. **在 vendored C++ binding 增加最小 server role** → verify：可以在 Domain 1/`lo` 发布带 CRC 的 HG LowState，
   不创建 LowCmd publisher，不依赖 `unitree_sdk2py` 或外部 checkout。
4. **新增 `G1MujocoDdsServer` 和显式 launcher** → verify：server 是独立进程，使用仓内 XML/limits；公开 registry
   仍只有 `g1/g1_real`，缺少 SDK 时报告明确且不影响普通直连 `g1`。
5. **先运行 state-only server，再启动只读 client harness** → verify：只在 Domain 1/`lo` 收到 HG LowState；
   `mode_machine=5`，29 motor slots、IMU、递增 tick 和 CRC 合同正确。
6. **执行隔离矩阵** → verify：`1/lo ↔ 1/lo` 可达；Domain mismatch、interface mismatch 均 readiness timeout，且
   没有 active LowCmd。
7. **验证独立性** → verify：临时移走/重命名 sibling `../unitree_mujoco` 后，build、tests、state-only server 和 client
   行为完全不变；运行时源码扫描也没有 import/path/subprocess 依赖。
8. **用官方 C++ simulator 做一次可选对照** → verify：两端 topic/type、字段、motor order、CRC/tick 语义一致；记录
   已知物理差异，不把外部闭环轨迹作为 golden。
9. **记录 state gap 并验证退出** → verify：保存最大/分位 gap、重复/丢失 tick 和 CPU load；client 先退出、server
   后退出，没有遗留 DDS participant 或后台进程。

### Phase 4 — Writer gate、watchdog 与仿真闭环

1. **实现原子 `arm(initial_target)`** → verify：`READY` 前调用失败；首次 active write 之前已有完整 target、gains、mode
   和时间戳，避免空 command 窗口。
2. **在 C++ writer 内执行 state/command watchdog** → verify：Python 卡住或 LowState 停止后，在批准的
   `timeout + control_dt` 上界内进入 fault；旧策略目标不再继续发布。
3. **校验所有 outgoing target** → verify：长度非 29、NaN/Inf、错误 motor mapping 均在覆盖旧 buffer 前失败并锁存 fault。
4. **按顺序启用第一方 simulator 控制** → verify：先保持当前姿态，再零速度 policy，最后小幅 x/y/yaw；29 个
   command slots 与仓内 XML actuator 顺序一一对应。
5. **证明直连/DDS 控制等价边界** → verify：相同模型状态、`dq=0/tau=0` 和相同 gains 下，DDS LowCmd 与直连
   `pd_target` 在每个物理子步产生相同的未裁剪/裁剪 torque；状态序列差异只能来自已记录的异步时序。
6. **保持本地 JoystickCtrl 作为第一阶段输入** → verify：轴方向、deadzone、A shutdown 与当前 `g1` golden 一致；
   暂不依赖模拟器是否正确填充 `wireless_remote[40]`。
7. **前置 stop/fault 检查** → verify：A、tilt、DDS fault 场景都在下一次 active `env.step()` 之前被拦截。
8. **修正 teardown 边界** → verify：构造、readiness、prepare、推理和主循环任意位置注入异常，均执行幂等 shutdown，
   不挂在 MotionSwitcher 或 writer join。

### Gate A — 允许完成 DDS 仿真验证的门槛

以下条件全部成立后才提交结果给用户再次确认：

- Domain/interface 正反矩阵和 optional-dependency 测试全部通过。
- CRC、tick、mode、freshness、29DoF 双向映射和四元数顺序有自动测试。
- state/command timeout 基于测量选定，fault latency 有上界和故障注入证据。
- 零命令、x/y/yaw、站立、跌倒、A shutdown 的手工 DDS 仿真记录完成。
- 无 active LowCmd-before-ready、stale target 重发或退出后 writer 残留。
- 第一方 server 在没有 `../unitree_mujoco` 的 checkout 中能独立 build、测试和运行；官方 simulator 仅有可选对照记录。
- 当前直连 `g1` 与第一方 DDS server 共用同一个 plant/XML/limits，没有第二份隐式机器人资产。
- 当前公开 `g1` 仍可按原命令运行，`g1_real` 未在自动测试中执行。

**默认授权边界止于 Gate A。** 未获得新的明确确认，不进入公开 `g1` 切换、旧 backend 删除或硬件 canary。

### Phase 5 — 可选：把公开 `g1` 切换到 DDS

1. **用两个固定 deployment profiles 替代 env 类型选择** → verify：`g1` 固定 `simulation/1/lo`，`g1_real` 固定
   `hardware/0/<required NIC>`；policy/checkpoint/joint contract 无复制。
2. **让两种配置共用 Unitree DDS environment** → verify：相同 LowState fixture 在 sim/real profile 产生相同 policy
   observation，相同 policy target 产生相同 29-slot LowCmd。
3. **保留输入差异** → verify：`g1` 仍为 local joystick，`g1_real` 仍为 remote；若以后统一输入，必须另测 remote bytes
   freshness、axes 和 A，不在本 cutover 顺手改变。
4. **保持 CLI 名称** → verify：`-c g1` 明确等待第一方 simulator readiness，`-c g1_real` 要求显式 `--net-if`；
   是否由 launcher 监督启动 server 必须只有一种明确行为，不自动选择网卡或 fallback。
5. **拆分生命周期字段** → verify：`prepare_enabled`、`safety_check` 和 target kind 各自只有一个职责，不再由
   `env.is_sim` 间接推断。
6. **更新架构与操作文档** → verify：所有示例都清楚表明 server/client 的启动和停止顺序；文档不把
   domain/watchdog/damping 写成硬件急停，也不要求下载外部 simulator。

### Gate B — 实机验证门槛

实机验证不是 Gate A 的自动延续。必须由用户另行明确授权，并满足 `docs/unitree_setup.md` 的独立急停、支撑、
隔离区和双人操作要求。顺序为：

1. **固定 checkout、binding wheel、SDK/Cyclone、checkpoint 和 resolved profile** → verify：归档 hash 和构建信息。
2. **先做 hardware state-only** → verify：Domain 0、显式非 `lo` NIC、CRC、递增 tick、mode、IMU 和 remote 数据正常，
   且没有 LowCmd writer。
3. **完成 operator gate 后受控 arm** → verify：初始姿态和零轴正确；任一 fault、异常动作或急停不可用立即中止。
4. **执行最短 canary** → verify：先零速度，再最小幅度单方向命令；不扩大范围，直到日志和人工检查均通过。

### Gate C — 删除旧直连 adapter 与外部参考依赖的门槛

只有第一方 DDS 仿真稳定、操作成本被接受、回滚版本完整且用户再次确认，才删除公开直连 backend 的装配代码：

- `DirectMujocoEnv` 的 registry/config preset，以及只服务直连路径的 pipeline 分支；
- 临时 client harness、迁移期兼容字段和只比较旧/新路径的 golden fixtures；
- 文档中要求 clone/build `../unitree_mujoco` 的操作步骤，只保留其固定 commit 作为历史对照记录。

以下内容是第一方 simulator 的运行闭包，**不能**随旧 adapter 删除：

- `G1MujocoPlant`、RoboJuDo 根依赖中的 `mujoco`；
- `assets/robots/g1/` 的 XML/meshes、torque/position limits；
- 第一方 `G1MujocoDdsServer`、server-role C++ binding 和安全测试；
- 若 server 继续提供 GUI，则保留 `third_party/mujoco_viewer/` 与可视化代码；若改为 headless，另立删除决策。

删除必须是独立 commit。最终 runtime、普通安装和 CI 中对 `../unitree_mujoco` 的路径、import、clone 或 subprocess
依赖必须为零。不得保留长期 feature flag 或两套公开默认实现来掩盖未完成的切换；回滚依赖明确 tag/commit。

## 6. 逐文件调整地图

| 文件/区域 | Phase | 计划变更 |
| --- | ---: | --- |
| `robojudo/environment/env_cfgs.py` | 1 | 加入 endpoint/domain 字段和 profile 交叉验证 |
| `robojudo/config/g1/env/g1_real_env_cfg.py` | 1 | real 固定 Domain 0、HG topics、identity 29DoF；移除通用 `eth0` 假设 |
| `robojudo/config/g1/g1_cfg.py` | 1/5 | 先显式 real endpoint；cutover 后改为两个原子 deployment profiles |
| `robojudo/simulation/g1_mujoco_plant.py`（新增） | 3 | 仓内 XML 的纯物理核心：state/reset/torque/step，可选 viewer；无 DDS/policy 依赖 |
| `robojudo/environment/mujoco_env.py` | 3/C | 迁移为 `G1MujocoPlant` 的薄直连 adapter；Gate C 后取消公开注册而非删除物理核心 |
| `robojudo/simulation/g1_mujoco_dds_server.py`（新增） | 3/4 | 独立 server orchestration、1 ms loop、LowCmd→torque、LowState snapshot |
| `robojudo/environment/unitree_cpp_env.py` | 2/4 | state-only、ready/arm/fault/status 生命周期和幂等 shutdown |
| `robojudo/controller/unitree_ctrl.py` | 4/5 | 保持 real remote；处理 status/stop，不把普通按钮冒充 deadman |
| `robojudo/pipeline/rl_pipeline.py` | 4/5 | fault/stop 前置、显式生命周期字段、异常传播 |
| `scripts/run_pipeline.py` | 4/5 | 构造与 prepare 纳入 `try/finally`；real NIC 必填；CLI 名称保持不变 |
| `third_party/unitree_cpp/src/unitree_controller.hpp` | 1/2/4 | domain、endpoint、status、phase、timestamp 和 thread-safe state |
| `third_party/unitree_cpp/src/unitree_controller.cpp` | 1/2/4 | 可配置 Init、coherent LowState、bounded release、writer gate/watchdog、幂等 stop |
| `third_party/unitree_cpp/src/py_binding.cpp` | 1/2 | domain parser 与 ready/arm/disarm/status API |
| `third_party/unitree_cpp/src/dds_sim_server.*`（新增） | 3/4 | 最小 HG server role、LowCmd CRC/快照、LowState CRC/publish；不包含 MuJoCo |
| `third_party/unitree_cpp/example/*` | 1/2 | 显式 endpoint 和正确生命周期示例 |
| `dds.md` 与阶段证据 | 每阶段 | 记录 vendored local changes、API 和重新构建要求；不得改写 pre-DDS 基线哈希 |
| `scripts/run_mujoco_dds_server.py`（新增） | 3/5 | 第一方 simulator 明确入口；只允许 simulation/1/lo |
| `scripts/run_dds_integration.py`（临时） | 3/4 | 不注册的只读/闭环 client harness；Gate C cutover 后删除 |
| `tests/test_dds_config.py`（新增） | 1 | profile、非法组合和 Unitree dict 合同 |
| `tests/test_dds_guard.py` / CTest（新增） | 2/4 | tick、freshness、phase、watchdog 和 failure output |
| `tests/test_dds_loopback.py`（新增） | 3/4 | subprocess mock peer；缺 SDK 时明确 skip |
| `tests/test_mujoco_plant.py`（新增） | 3/4 | 直连/DDS torque golden、state/reset、XML actuator order 和 finite-state 合同 |
| `tests/test_dds_sim_independence.py`（新增） | 3/C | 禁止运行时引用 sibling `unitree_mujoco`，验证仓内 asset/server 闭包 |
| `tests/test_pipeline_safety.py`（新增） | 4 | A/tilt/DDS fault 在 active target 前生效 |
| `tests/test_full_imports.py` | 1/5/C | 固定 profile、vendor/API、最终 environment/asset closure |
| `README.md`, `docs/*.md`, `AGENTS.md` | 5/C | 同步实际流程、安全边界和验证能力 |

配置 Registry/ConfigManager 的整体扁平化仍是独立重构。DDS 迁移只提供简单的 endpoint/profile 数据结构，不能用
新的 factory/manager 层替换旧 Registry；否则只是增加 enterprise sludge，无法判断回归来自通信还是配置装配。

## 7. 验证矩阵

| 场景 | 预期结果 |
| --- | --- |
| simulator `1/lo` + client `1/lo` | READY，允许人工 gate 后 arm |
| sibling `../unitree_mujoco` 不存在 | 第一方 server build/tests/runtime 完全不受影响 |
| simulator `1/lo` + client `0/lo` | readiness timeout；零 active LowCmd |
| simulator `1/lo` + client `1/NIC` | 明确不可达/初始化失败；零 active LowCmd |
| hardware profile + `lo` | 创建 DDS 前配置错误 |
| simulation profile + Domain 0 | 创建 DDS 前配置错误 |
| CRC 错误持续到达 | 不刷新 freshness，最终 fault |
| tick freeze/duplicate | 不刷新 freshness，最终 fault |
| uint32 tick 正常 wrap | 不 fault |
| tick regression / mode change | fault 锁存，禁止继续 active target |
| 未 READY/未 arm 调用 `step()` | 拒绝，零 active publish |
| target 长度错误、NaN 或 Inf | 拒绝且不污染旧 command buffer |
| Python inference 停止 | command watchdog 在有证据的上界内触发 |
| LowState 停止 | state watchdog 在有证据的上界内触发 |
| A、tilt、DDS fault | 下一次 active env step 前被拦截 |
| 重复 shutdown | 幂等，不恢复 ACTIVE，不遗留 writer |
| 同进程请求第二 endpoint | 明确失败 |
| 两个 LowCmd publisher | 不支持；preflight/进程约束必须发现或禁止 |

测试分层：

- **L0 普通 CI：** 配置、29DoF、DoFAdapter、观测、CLI、fake binding、pipeline 调用顺序。
- **L1 C++ 单测：** tick 分类、状态机、时间戳、watchdog；注入时钟，不用 `sleep()`。
- **L2 DDS loopback：** 独立 subprocess mock peer，Domain 1/`lo`，验证 HG state/command/CRC。
- **L3 第一方 MuJoCo DDS：** 仓内 XML/plant/server 的独立 subprocess 集成与故障注入。
- **L3-reference 可选对照：** 固定 `unitree_mujoco` commit，只比较 wire contract，不进入 release gate 的运行依赖。
- **L4 实机：** 永不进入 CI，仅在明确授权和完整硬件 preflight 后执行。

基础验证命令计划为：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check robojudo scripts tests
ruff format --check robojudo scripts tests
cmake -S third_party/unitree_cpp -B /tmp/robojudo-unitree-build -DBUILD_TESTING=ON
cmake --build /tmp/robojudo-unitree-build --parallel
ctest --test-dir /tmp/robojudo-unitree-build --output-on-failure
git diff --check
```

## 8. 中止条件、兼容与回滚

任一情况立即中止当前阶段，不允许 fallback：

- endpoint 与 profile 不一致、接口不存在或意外发现实机 participant；
- CRC 持续失败、tick 不前进/回退、mode 改变、state/command stale；
- 关节映射、四元数、轴方向、gains 或 50 Hz 合同不一致；
- 模拟器退出、writer 未停止、两个 LowCmd publisher、进程无法干净 teardown；
- hardware E-stop、支撑、隔离区、双人操作或 operator authorization 任一不满足。

回滚策略：

1. domain plumbing、readiness/status、watchdog、DDS harness、公开 cutover、旧 backend 删除分别提交。
2. Gate A 前公开 `g1` 不变；Gate C 前旧直连 adapter 与新 DDS server 并存并共享同一个 MuJoCo plant。
3. 保存 `pre-dds` tag、已验证 Python environment、UnitreeCpp wheel、SDK/Cyclone 版本和 resolved config。
4. C++ ABI 变化时 Python、binding wheel、SDK/Cyclone 必须整体回滚并重建，禁止混用新旧组件。
5. 第一方 simulator 的 plant、XML、binding 和 client 必须作为同一兼容集合回滚；外部 `unitree_mujoco` commit
   只影响可选 L3-reference 对照，不得影响 release runtime。

兼容承诺：

- Gate A 前，`-c g1`、`-c g1_real`、checkpoint、29DoF 数值和默认安装行为不变。
- Phase 5 若执行，CLI 名称保持，但 `-c g1` 将要求第一方 simulator 进程；启动由同一 launcher 监督还是两个终端
  显式执行必须在 cutover 前固定并记录，不能依赖外部 checkout。
- `g1_real` 永远不是默认配置，永远不作为自动测试命令。

## 9. 待确认事项

建议本轮只授权实施到 **Gate A**，采用以下默认选择：

1. 保留公开进程内 `g1`，DDS 仿真使用临时、不注册的 harness。
2. Phase 3 起实现第一方 `G1MujocoPlant + G1MujocoDdsServer`；最终 build/runtime/CI 不依赖外部
   `unitree_mujoco`、其模型或路径。
3. 外部 C++ `unitree_mujoco@ae6a8403e272` 只做一次可选 wire-contract 对照，不修改、不 vendor、不建立
   Git submodule，也不作为 Gate A 的运行前置条件。
4. DDS 仿真第一阶段继续使用 RoboJuDo local joystick；输入路径统一另行评审。
5. Domain 固定为 sim=1、real=0；sim 固定 `lo`；real NIC 启动时显式必填。
6. 先测量 gap 再决定 watchdog timeout；failure output 需单独安全评审。
7. DDS 迁移期间不同时引入 Hydra 或进行完整 Registry/ConfigManager 重构。
8. Gate A 后提交证据并暂停；Phase 5、Gate B 和 Gate C 均需新的明确授权。

确认建议回复：

```text
确认按 dds.md 执行至 Gate A；第一方实现 DDS simulator，外部 unitree_mujoco 仅作可选对照；
不切换公开 g1，不运行实机，不删除旧直连 adapter。
```
