# 稳定输入

本路线的基础仿真和小尺寸联合仿真不读取实测数据。32×32 联合虚拟阵列默认使用包资源 [reference_element_geometry.csv](../src/array_reconstruct/data/reference_element_geometry.csv)，共 1,024 行、53,626 字节，SHA-256 为 `d9553b6f0d1eaaa1a2c0228b6b568841abc85de5658a82befbffcb10a327bfa3`。

2026-11-12发布整理将CSV和原始来源JSON纳入普通wheel，通过 `importlib.resources` 定位，安装后不依赖源码树。`metadata/` 下的[兼容快照](../metadata/reference_element_geometry.csv)和[来源说明](../metadata/reference_element_geometry_provenance.json)保留迁入字节，回归测试核对包资源与快照一致。显式传入几何路径仍受支持。其中旧 `temp/` 和归档路径是历史来源标识，不参与默认资源查找。

此表仅作为假设几何和编号参考，不能据此确认实物、测量单元身份或几何。历史大实测、算子缓存及运行结果仍在旧项目；完整恢复位置和哈希见 [source-manifest.json](../source-manifest.json)。
