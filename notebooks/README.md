# Qlib 量化研究详细教程 - 模块化版本

本目录包含了将原始 `detailed_workflow.ipynb` 拆分后的模块化notebook文件，每个模块专注于特定的功能领域，便于学习和使用。

## 文件结构

```
notebooks/
├── 01_introduction_and_config.ipynb    # 介绍与配置模块
├── 02_data_module.ipynb                # 数据模块
├── 03_model_training.ipynb             # 模型训练模块
├── 04_evaluation_module.ipynb          # 评估模块
├── 05_utils_and_helpers.ipynb          # 工具函数与辅助模块
└── README.md                           # 本说明文件
```

## 模块说明

### 1. 介绍与配置模块 (`01_introduction_and_config.ipynb`)
- **功能**：项目介绍、基本配置设置
- **内容**：
  - 教程简介
  - 基本参数配置（市场、基准、实验名称等）
  - 必要的导入语句

### 2. 数据模块 (`02_data_module.ipynb`)
- **功能**：数据获取、预处理、数据集构建
- **内容**：
  - 数据下载与初始化
  - 原始数据检查（日历、OHLCV、复权等）
  - 静态vs动态股票池
  - Point-In-Time数据
  - 数据加载器（DataLoader）
  - 数据处理器（DataHandler）
  - 数据集构建（DatasetH、TSDatasetH）
  - 现成数据集使用（Alpha158）

### 3. 模型训练模块 (`03_model_training.ipynb`)
- **功能**：模型训练、推理、信号生成
- **内容**：
  - 模型接口介绍
  - 模型配置与初始化
  - 数据集准备
  - 模型训练流程
  - 信号生成与保存

### 4. 评估模块 (`04_evaluation_module.ipynb`)
- **功能**：回测、分析、报告生成
- **内容**：
  - 回测配置与执行
  - 信号分析
  - 投资组合分析
  - 风险分析
  - 模型性能评估
  - 结果可视化

### 5. 工具函数与辅助模块 (`05_utils_and_helpers.ipynb`)
- **功能**：常用工具函数、辅助功能
- **内容**：
  - 收益率计算工具
  - 数据可视化工具
  - 性能优化技巧
  - 调试和错误处理
  - 配置管理

## 使用建议

### 学习顺序
1. **初学者**：按顺序学习 01 → 02 → 03 → 04 → 05
2. **有经验用户**：可以直接查看感兴趣的模块
3. **工具参考**：05模块作为工具参考，随时查阅

### 运行环境
- Python 3.7+
- Qlib 最新版本
- 相关依赖包（pandas, numpy, matplotlib, plotly等）

### 注意事项
1. 每个模块都可以独立运行，但建议按顺序学习
2. 某些模块之间需要传递参数（如实验ID），请参考代码注释
3. 数据下载可能需要较长时间，请耐心等待
4. 建议在Jupyter环境中运行，以便查看图表和交互式结果

## 原始文件
原始的 `detailed_workflow.ipynb` 文件仍然保留，包含完整的流程演示。模块化版本更适合：
- 学习和理解特定功能
- 作为参考文档
- 快速查找特定实现
- 教学和培训

## 贡献
如果您发现任何问题或有改进建议，欢迎提交Issue或Pull Request。

## 许可证
Copyright (c) Microsoft Corporation. Licensed under the MIT License.

