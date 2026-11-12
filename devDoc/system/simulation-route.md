# 仿真路线实现与迁移验证

当前 active 路线是 [experiments/20261027-simulation](../../experiments/20261027-simulation/README.md)。它保留旧实现，尚未优化；迁移本身不构成新的算法任务、仿真运行或真实反演授权。

## 实现边界

基础 `simulation.py` 仅依赖 NumPy，可独立生成虚拟近场；`physics/inverse/sparse/validation` 及联合几何、有限孔径和姿态模块形成自包含数值依赖闭包。小尺寸联合虚拟阵列使用公式；32×32 联合虚拟阵列读取随包分发的参考几何 CSV。基础模拟器与求解器均不需要旧 `.memory`、测量归档或历史 `outputs/`。

迁入 34 个数值模块，原始内容保持字节一致；另外保留原 `cli.py`。唯一修改的原文件是 `pyproject.toml`（仿真入口与可选 torch 依赖）和 `src/array_reconstruct/__main__.py`（入口导入）；新增 `simulation_cli.py` 仅限制命令集合，再委派给原 CLI。未修改算法、数值门限、核、seed 或配置内容。原/新 SHA-256 由 [source-manifest.json](../../experiments/20261027-simulation/source-manifest.json)逐文件给出。

原 `cli.py` 含历史 `analyze-real` 懒导入，所需 `real_measurement.py` 未迁入。应使用发布的 `python -m array_reconstruct` 或安装后的 `array-reconstruct`，它们仅列出四个仿真命令。

## 输入与输出路径

几何CSV及来源JSON现位于 `src/array_reconstruct/data/`，由 `importlib.resources` 定位，支持普通wheel和editable安装。`metadata/` 保留原字节兼容快照；回归测试核包资源与快照一致，1024阵元内容和顺序不变。稳定输入说明从实验目录 `data/README.md` 导航。参考几何仅是假设与编号配置，不证明实物或实测身份。

运行命令以实验目录为工作目录，`src/` 通过安装或 `PYTHONPATH` 定位。一次性检查或未来试运行显式把输出放在项目根 `tmp/<task>/`。旧算法默认的 `outputs/` 位置保持原文；历史策略中的输出路径、旧 seed 和来源描述也保持原文，未复制实际输出。

`thesis_calibration`、`thesis_coded` 的完整历史脚本需要原算子缓存 NPZ，留在旧项目；后续若确需复现，显式设置 `--operators` 和 `--output`，并按原策略恢复输入。对应数值 API 和迁入的维护测试是独立的。`thesis_coded` 运行时会哈希它的脚本与测试文件，二者已经原字节迁入。

## 来源、环境和排除

来源根目录为 `D:\project\trifles\20261027 array reconstruct`。检查 `git rev-parse`、`status`、`log` 均报告该位置不是 Git 仓库，故不存在可继承的源码提交历史；采用逐文件 SHA-256。目标路线复制前为空，没有覆盖现有路线文件。源内容保持只读，未移动或删除任何源文件。

源 `src/tests/scripts/configs` 共 233 个非缓存文件、4,684,123 字节，全部在 manifest 中记录保留/排除、源路径和 SHA-256。选定复制 77 个文件，原始合计 1,108,138 字节；其中 75 个原字节保留，2 个仅做入口/声明适配。README、输入导航和新入口属于新增迁移文件。

未迁入的代码分为历史真实反演/调查、源注册、调度/确认执行、staged/training/qualification 审计、warm certificate 和 nearbound 恢复链。`.memory/`、历史 `outputs/`、大 `sources/`、`temp/`、资源导入包和缓存留在旧路径；聊天和结论的项目级迁移由根文档另行索引。完整文件清单与分组原因见 manifest。

2026-11-10 检查使用本机 `C:\Program Files\Python313\python.exe`，Python 3.13.17；NumPy 2.5.4、SciPy 1.18.0、Matplotlib 3.10.9、pytest 9.2.0、torch 2.10.0+cu128、setuptools 70.2.0。该环境已存在，未创建虚拟环境、安装依赖、下载模型或移动共享环境。旧 requirements 的固定 NumPy/pytest 版本与本次现有环境不同，保留为历史版本参考，未声称在旧精确环境重验。

源活动进程命令行查询受本机 CIM 访问限制；该项未获得完整命令行证据。迁移只复制且不修改源和既有输出，因此不依赖停止旧进程。

## 当前稳定K4首版（2026-11-12）

用户已要求执行研究计划。本轮预登记16次机制开发及20例独立确认，原数值模块不变；稳定轻档K4在11.7 GHz的健康/5故障共20新案全部通过0.5 dB／5°／10%系数基准，最坏0.166144 dB／1.969274°／3.759260%。范围固定为正常幅度均匀±2.5 dB、相位σ5°、非参考码未知高斯换码误差σ0.2 dB／3°且跨平面固定；会话漂移、未知几何和其他频率未确认。检测另有6次正常误报，不宣称全部诊断达标。

新增 `src/array_reconstruct/k4_research.py`、`scripts/run_k4_research.py`、`tests/test_k4_research.py` 和 `configs/k4_development_20261112.json` / `k4_confirmation_20261112.json`。新运行器保留开发/确认用途、冻结策略、按验证选alpha、完整案例分母、固定绝对噪声、对齐与未对齐指标、掩膜计数、输入与源码哈希、预算及逐案原证。它复用原 `thesis_coded.prepare_coded_ridge/solve_coded_ridge`，普通 `reconstruct` 入口仍是单状态。

全规模运行使用原缓存只读输入，其SHA-256与历史回执一致；新入口可显式 `--generate-operators` 重建，无需旧项目。当前运行命令、全规模生成核对、输入/输出细节及限制统一见[首版结果报告](../tasks/20261112-simulation-research/research/k4-first-version-results.md)，简要命令从[实验README](../../experiments/20261027-simulation/README.md)进入。历史来源清单继续记录迁移时身份，不将本次新增文件冒充原字节迁入。

科学运行时新增运行器25项、原编码/大离散41项及逆解/物理36项，共102项相关CPU检查通过；生成算子的微型端到端烟测通过。36案的评分、固定噪声、源码/策略/档案哈希独立复核无问题，研究合同及结论经独立审阅。它们与下述迁移时441项检查分别保留，不合并成新增测试数。选留新原证在 `artifacts/runs/k4-research-20261112/`；随后发布整理见下节。

## 首版发布整理（2026-11-12）

用户要求调查后在可行时提交GitHub。发布前修复普通wheel缺少参考几何和K4回执依赖包外启动脚本两处安装问题：几何资源内容不变；源码树仍哈希启动脚本，已安装包则在 `unavailable_sources` 明确记录该脚本未分发，其余模块继续真实哈希。数值算法、策略、选参和评分均未改动。

实际普通wheel与editable安装、树外几何/CLI/K4 API检查通过；CPU完整维护检查469项、额外可选张量CPU接口43项通过，未执行CUDA。正式确认策略的生成算子完整复现及公开仓库状态见[发布记录](../archive/20261112-github-publication/result.md)。原确认使用的冻结K4源码另随小证据发布，原回执哈希不替换为本轮工程修正版。

## 已承接的旧科学结果（2026-11-12）

旧科学进度并非从零：小误差稀疏验证、大离散54案例/216拟合、区域提取、编码及闭环均有迁入报告。K4理想/轻档状态误差在旧6确认案例有条件性阳性，强档失败；原单态大离散和区域方法未过原系数门，闭环一致性收益不能代替恢复准确。现有固定条件下的数值资格也已按原适用范围保留。详见[虚拟成果承接表](../tasks/20261112-simulation-research/research/prior-virtual-results.md)和[联合/数值承接表](../tasks/20261112-simulation-research/research/prior-joint-results.md)。

11份顶层报告均已迁入；原53件小原证复核一致，另补8份小摘要/逐案指标，见[更新后的证据索引](../archive/legacy-20261027/evidence-index.md)。基础API、编码/闭环/区域实现及其维护测试已经保留；历史完整运行器所需NPZ缓存仍是单独输入依赖，不代表这些实验从未实施。当前计划据已有K4候选设计增量工作，不因迁移重新开展全部旧实验；本节不增加新的科学通过声明。

## 迁移时实际验证（2026-11-10）

验证运行日志和临时产物放在根目录 `tmp/migration-code/`，可清理；稳定结论在本节与 manifest 保存。

- 71 个迁入/新增 Python 文件通过内存编译，不在源码目录生成 pyc。
- 35 个模块在现有环境完成导入；包括可选张量/GPU模块的导入，未运行 CUDA。
- 路线入口、`simulate --help`、`reconstruct --help` 通过，只列出仿真命令。
- 已有纯 CPU 维护测试：441 项通过，用时 23.41 秒。2 个 RuntimeWarning 来自同一非有限数防护用例的刻意异常输入，测试通过；全局 pytest-asyncio 插件另提示未设置默认 loop scope，不影响本次同步测试。
- 5 个可选 tensor/CUDA 测试文件未执行：`test_joint_amplitude_gpu.py`、`test_joint_geometry_gpu.py`、`test_joint_geometry_gpu_amplitude.py`、`test_joint_pose_gpu_interface.py`、`test_joint_finite_aperture_backend.py`。源测试字节与默认发现规则保持不变。

本次未执行完整验证矩阵、生成新的仿真实验结果、重跑旧实验、读取大实测归档或进行真实反演。轻量检查证明本机的源码入口与维护测试可用，不构成 GPU 数值/性能或科学准确性的新增结论。
