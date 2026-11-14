# 英文README与GitHub About

2026-11-14，Asia/Shanghai。按用户要求联网参考技术写作规范，更新GitHub About，并将根README改为英文。About实际写入并回读验证为：

> Recover relative antenna array element amplitudes and phases from simulated near-field measurements.

参考[GitHub README指南](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)、[Google技术写作语气指南](https://developers.google.com/style/tone)，并查看[pyArgus](https://github.com/petotamas/pyArgus)和[nf2ff](https://github.com/saaaaamuel/nf2ff)的功能介绍。以项目功能、复现入口和验证条件组织内容，避免宣传性形容词、空泛套话和未经证据支持的能力声明；没有仿写其他项目的正文。

根README保留K4方法和有限仿真范围，补充Bash/zsh与PowerShell命令，明确10个独立种子配对形成20案、逐项最坏值、公共复标量评分对齐及故障诊断局限；说明现有图视频和支持文档仍使用中文。原始科学证据和算法未改动。

独立科学/英文只读审阅通过；12个本地链接均指向已跟踪文件，两份策略路径、模块与K4脚本help及差异空白检查通过。仅文档和仓库description修改，没有重跑科学案例或把历史469/43项检查记成本次新测试。

任务启动的memory brief因本机数据库连接拒绝而失败，直接核对现行代码、结果报告和已提交记录后继续；短任务未启动自动维护。临时验证和About回读收据位于 `tmp/english-readme-about-20261114/`，不进入Git。
