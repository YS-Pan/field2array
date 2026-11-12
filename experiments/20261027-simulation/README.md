# 20261027 simulation

本项目唯一 active 路线。这里是从 `D:\project\trifles\20261027 array reconstruct` 迁入的仿真与数值实现，保留来源日期。2026-11-12按[当前研究计划](../../devDoc/tasks/20261112-simulation-research/plan.md)完成16次开发和20例新确认，稳定轻档K4首版全部通过原系数三门；[范围和结果](../../devDoc/tasks/20261112-simulation-research/research/k4-first-version-results.md)限定11.7 GHz及稳定状态。原求解器保持不变，新增策略化运行入口。

当前实现、迁移边界及验证结论统一维护在 [devDoc/system/simulation-route.md](../../devDoc/system/simulation-route.md)。逐文件来源、SHA-256、适配项和排除清单见 [source-manifest.json](source-manifest.json)。

## 入口

从本实验目录执行 PowerShell 命令；可使用已具备依赖的 Python，无需先创建环境：

```powershell
Set-Location 'experiments/20261027-simulation'
$env:PYTHONPATH = Join-Path $PWD 'src'
python -B -m array_reconstruct --help
python -B -m array_reconstruct simulate --help
python -B -m array_reconstruct reconstruct --help
```

发布的入口仅提供 `simulate`、`reconstruct`、`validate`、`validate-sparse`，调用原有实现。将来开展获授权的仿真时，显式使用项目根 `tmp/<task>/` 输出，例如 `--output ../../tmp/<task>/virtual_measurement.npz`；原始实现的默认 `outputs/` 路径未被重写。

稳定K4首版复现从**项目根**执行（输出目录须新建；生成算子无需旧项目）：

```powershell
python -B 'experiments/20261027-simulation/scripts/run_k4_research.py' `
  --policy 'experiments/20261027-simulation/configs/k4_confirmation_20261112.json' `
  --generate-operators --output 'tmp/k4-confirmation-reproduction'
```

已有兼容纯仿真算子可用 `--operators '<绝对NPZ路径>'` 替代 `--generate-operators`；输入几何、物理核和哈希会核验。开发使用 `k4_development_20261112.json` 与新输出目录。该脚本进行科学复现，直接四码求解使用 `thesis_coded.prepare_coded_ridge/solve_coded_ridge` API；普通 `reconstruct` 仍是单状态入口。完整合同、预算、102项相关检查和限制统一见根开发文档。

如需在已有环境安装该路线，使用 `python -m pip install -e '.[test]'`；普通安装使用 `python -m pip install .`，所需32×32几何已作为包资源提供。可选张量/GPU 实现另需 `.[gpu]`。发布前实际安装检查见[发布记录](../../devDoc/archive/20261112-github-publication/result.md)。原 [requirements-numerics.txt](requirements-numerics.txt) 是旧验证环境的版本记录，原文中 memory 独立 lock 指向旧项目，不代表本项目运行着旧 memory。

## 保留范围

- `src/`：34 个原字节仿真/数值模块，以及原 `cli.py`、适配后的 `__main__.py` 和新增 `simulation_cli.py`。包括物理核、独立虚拟生成、逆解/稀疏、联合几何/姿态、有限孔径、CPU 和可选张量/GPU实现。
- `tests/`：28 个原字节维护测试文件。默认 pytest 发现规则与旧包相同。
- `configs/`：3 个原字节纯仿真策略；它们的旧结果位置、历史 seeds 和门限是来源记录，不代表新的运行授权或完成证据。
- `scripts/`：6 个原字节纯仿真/数值控制入口；本次只迁入，未运行完整实验。
- [稳定几何输入](data/README.md)：53,626 字节 CSV 及其原始来源 JSON，保留 `metadata/` 兼容路径，避免修改算法源码。

迁移时 `pyproject.toml` 仅调整命令入口并声明可选 `torch`；`__main__.py` 仅调整导入入口。新增 `simulation_cli.py` 仅做命令分发，所有数值函数继续使用原实现。2026-11-12发布整理另修正参考几何的包资源定位和K4安装环境的来源回执，数值算法与几何内容不变；原迁移统计和来源清单保留当时身份。原 `cli.py` 保留来源字节，它的历史 `analyze-real` 分支需要未迁入的实测模块，不能作为当前路线入口直接使用。

## 维护检查

纯 CPU 检查命令如下。被显式排除的 5 个文件需要可选 torch 或 CUDA；其原文件已保留，日后可按环境直接执行。

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp ../../tmp/migration-code/pytest `
  --ignore=tests/test_joint_amplitude_gpu.py `
  --ignore=tests/test_joint_geometry_gpu.py `
  --ignore=tests/test_joint_geometry_gpu_amplitude.py `
  --ignore=tests/test_joint_pose_gpu_interface.py `
  --ignore=tests/test_joint_finite_aperture_backend.py
```

完整 `python -m pytest` 会按原测试规则发现这些可选检查；本次没有执行张量/CUDA检查或历史完整实验。迁移实际结果以根目录的[验证记录](../../devDoc/system/simulation-route.md)为准。

## 历史路径

真实测量、旧调查/绘图/交付、调度/续跑/封存状态、旧 memory 和大数据留在旧项目。源码清单逐项记录未迁入的 `src/tests/scripts/configs` 文件及 SHA-256；其他大目录按来源组记录，未复制。

`thesis_calibration.py` 和 `thesis_coded.py` 的完整历史运行器需要调用者显式提供旧算子 NPZ（脚本参数 `--operators`），其数值函数和维护测试不依赖该缓存。旧 `run_joint_geometry_virtual.py` 已混合 GPU 收据、预算、续跑和锁状态，留在旧目录；当前模块 API 不依赖它。迁移未携带旧授权、旧完成状态或旧自动运行规则。
