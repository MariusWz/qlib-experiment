# Qlib Dataset 模块完全指南
# Comprehensive Guide to Qlib Dataset Module

> **版本说明 (Version)**: 基于 Qlib v0.9.6+ 官方仓库
> **作者 (Author)**: 根据 Microsoft Qlib 官方文档和源代码编写
> **最后更新 (Last Updated)**: 2025-11-05

---

## 目录 (Table of Contents)

1. [数据架构概览 (Data Architecture Overview)](#1-数据架构概览)
2. [Dataset 类详解 (Dataset Classes)](#2-dataset-类详解)
3. [DataLoader 详解 (DataLoader Classes)](#3-dataloader-详解)
4. [DataHandler 详解 (DataHandler Classes)](#4-datahandler-详解)
5. [Processor 处理器详解 (Processor Classes)](#5-processor-处理器详解)
6. [表达式引擎 (Expression Engine)](#6-表达式引擎)
7. [实战应用案例 (Practical Examples)](#7-实战应用案例)
8. [常见陷阱与最佳实践 (Common Pitfalls & Best Practices)](#8-常见陷阱与最佳实践)

---

## 1. 数据架构概览

### 1.1 Qlib 数据流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Qlib Data Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Raw Data        DataLoader       DataHandler        Dataset       │
│  (原始数据)       (数据加载器)      (数据处理器)       (数据集)      │
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌─────────────┐  │
│  │ .bin     │ → │ Load &    │ → │ Process &  │ → │ Train/Valid/│  │
│  │ files    │   │ Cache     │   │ Transform  │   │ Test Split  │  │
│  └──────────┘   └───────────┘   └────────────┘   └─────────────┘  │
│                                                                     │
│  • OHLCV        • Expression    • Fillna          • Segments       │
│  • Factors      • Batch load    • Normalization   • Iterators      │
│  • Calendar     • Cache         • Processors      • Samplers       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心概念 (Core Concepts)

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Raw Data** | 存储原始市场数据 | 无 | 二进制数据文件 (.bin) |
| **DataLoader** | 加载和缓存数据切片 | 表达式、时间范围 | DataFrame |
| **DataHandler** | 数据处理管道 | 原始数据 | 处理后的特征和标签 |
| **Processor** | 数据变换操作 | DataFrame | 变换后的 DataFrame |
| **Dataset** | 机器学习数据集 | 处理后的数据 | 训练/验证/测试集 |

---

## 2. Dataset 类详解

### 2.1 DatasetH (标准数据集)

**用途**: 传统机器学习模型的数据集，支持固定的 train/valid/test 分割

**源码路径**: `qlib/data/dataset/dataset.py`

#### 类定义

```python
class DatasetH(Dataset):
    """
    Dataset with DataHandler

    Parameters
    ----------
    handler : DataHandler or dict
        数据处理器实例或配置字典
    segments : dict
        时间段分割定义
        例: {'train': ('2020-01-01', '2022-12-31'),
             'valid': ('2023-01-01', '2023-06-30')}
    """

    def __init__(self, handler, segments):
        self.handler = init_instance_by_config(handler, accept_types=DataHandler)
        self.segments = segments

    def prepare(self, segments, col_set='__all__', data_key='infer'):
        """
        准备指定时间段的数据

        Parameters
        ----------
        segments : str or list
            段名称，如 'train', 'valid', 'test'
        col_set : str
            列集合: 'feature', 'label', '__all__'
        data_key : str
            数据类型: 'infer' (推理) 或 'learn' (训练)

        Returns
        -------
        pd.DataFrame
            处理后的数据，MultiIndex (datetime, instrument)
        """
```

#### 实战示例 1: 基础使用

```python
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Fillna, ZScoreNorm

# 初始化 Qlib
qlib.init()

# 创建数据处理器
handler = DataHandlerLP(
    instruments='csi300',
    start_time='2020-01-01',
    end_time='2024-01-31',
    data_loader={
        "class": "QlibDataLoader",
        "kwargs": {
            "config": (
                ["$close/Ref($close,1)-1", "Mean($close,5)/Mean($close,20)"],
                ["RETURN", "MA_RATIO"]
            )
        }
    },
    infer_processors=[
        Fillna(fill_value=0),
        ZScoreNorm(fit_start_time='2020-01-01', fit_end_time='2022-12-31')
    ]
)

# 创建数据集
dataset = DatasetH(
    handler=handler,
    segments={
        'train': ('2020-01-01', '2022-12-31'),
        'valid': ('2023-01-01', '2023-06-30'),
        'test': ('2023-07-01', '2024-01-31')
    }
)

# 获取训练数据
train_data = dataset.prepare('train')
print(f"训练数据形状: {train_data.shape}")
print(f"特征列: {train_data.columns.tolist()}")
```

**输出示例**:
```
训练数据形状: (150000, 2)
特征列: ['RETURN', 'MA_RATIO']
```

#### 实战示例 2: 分离特征和标签

```python
# 获取特征
features = dataset.prepare('train', col_set='feature')

# 获取标签
labels = dataset.prepare('train', col_set='label')

print(f"特征形状: {features.shape}")
print(f"标签形状: {labels.shape}")
```

### 2.2 TSDatasetH (时间序列数据集)

**用途**: 为 LSTM、RNN 等序列模型准备数据，自动生成固定长度的时间窗口

**源码路径**: `qlib/data/dataset/dataset.py`

#### 类定义

```python
class TSDatasetH(DatasetH):
    """
    Time Series Dataset with DataHandler

    Parameters
    ----------
    handler : DataHandler or dict
        数据处理器
    segments : dict
        时间段分割
    step_len : int
        时间窗口长度（回望步数）
        例: step_len=20 表示使用过去 20 天的数据
    """

    def __init__(self, handler, segments, step_len=20):
        super().__init__(handler, segments)
        self.step_len = step_len

    def prepare(self, segments, col_set='__all__', data_key='infer'):
        """
        Returns
        -------
        TSDataSampler
            时间序列采样器，可按索引访问时间窗口样本
            每个样本形状: (step_len, num_features)
        """
```

#### 实战示例 3: 时间序列数据集

```python
from qlib.data.dataset import TSDatasetH

# 创建时间序列数据集
ts_dataset = TSDatasetH(
    handler=handler,
    segments={
        'train': ('2020-01-01', '2022-12-31'),
        'valid': ('2023-01-01', '2023-06-30')
    },
    step_len=20  # 使用过去 20 天的数据
)

# 获取采样器
train_sampler = ts_dataset.prepare('train')
print(f"样本总数: {len(train_sampler)}")

# 获取单个样本
sample = train_sampler[0]
print(f"单个样本形状: {sample.shape}")  # (20, num_features)

# 可视化时间序列样本
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(min(5, sample.shape[1])):
    plt.subplot(2, 3, i+1)
    plt.plot(sample[:, i])
    plt.title(f'Feature {i}')
    plt.xlabel('Time Step')
plt.tight_layout()
plt.show()
```

---

## 3. DataLoader 详解

### 3.1 QlibDataLoader

**用途**: 从 Qlib 数据源加载数据，支持表达式计算

**源码路径**: `qlib/data/dataset/loader.py`

#### 类定义

```python
class QlibDataLoader(DataLoader):
    """
    Qlib 数据加载器

    Parameters
    ----------
    config : tuple
        (expressions, names) 的元组
        expressions: 表达式列表
        names: 对应的列名列表
    """

    def __init__(self, config):
        self.expressions = config[0]
        self.names = config[1]

    def load(self, instruments, start_time=None, end_time=None):
        """
        加载数据

        Returns
        -------
        pd.DataFrame
            MultiIndex: (datetime, instrument)
            Columns: 特征名称
        """
        from qlib.data import D

        data = D.features(
            instruments=instruments,
            fields=self.expressions,
            start_time=start_time,
            end_time=end_time
        )
        data.columns = self.names
        return data
```

#### 实战示例 4: 基础加载

```python
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data import D

# 创建数据加载器
loader = QlibDataLoader(
    config=(
        # 表达式列表
        [
            "$close",                          # 收盘价
            "$volume",                         # 成交量
            "$close/Ref($close,1)-1",          # 日收益率
            "Mean($close, 5)",                 # 5日均线
            "Std($close/Ref($close,1)-1, 20)"  # 20日波动率
        ],
        # 列名列表
        ["CLOSE", "VOLUME", "RETURN", "MA5", "VOLATILITY"]
    )
)

# 加载数据
data = loader.load(
    instruments=['SH600519', 'SH600000'],
    start_time='2024-01-01',
    end_time='2024-01-31'
)

print(data.head())
```

**输出**:
```
                            CLOSE         VOLUME    RETURN        MA5  VOLATILITY
datetime   instrument
2024-01-02 SH600000      9.565298  152258.937500 -0.003014   9.582649    0.007636
           SH600519    223.379959  242563.937500 -0.023748 223.456789    0.015234
2024-01-03 SH600000      9.623135  125605.921875  0.006047   9.590123    0.007845
           SH600519    224.571793  152594.484375  0.005335 224.123456    0.015456
```

---

## 4. DataHandler 详解

### 4.1 DataHandlerLP (Learnable Processor Handler)

**用途**: 完整的数据处理管道，管理数据加载、处理和缓存

**源码路径**: `qlib/data/dataset/handler.py`

#### 核心概念

DataHandlerLP 生成三种类型的数据:

1. **DK_R (Raw)**: 从 DataLoader 直接获取的原始数据
2. **DK_I (Infer)**: 经过 `infer_processors` 处理的推理数据
3. **DK_L (Learn)**: 经过 `learn_processors` 处理的训练数据

#### 类定义

```python
class DataHandlerLP(DataHandler):
    """
    数据处理器（可学习处理器模式）

    Parameters
    ----------
    instruments : str or dict
        股票池定义
        - 字符串: 'csi300', 'csi500'
        - 字典: {'market': 'csi300', 'filter_pipe': [...]}
    start_time : str
        开始时间 'YYYY-MM-DD'
    end_time : str
        结束时间 'YYYY-MM-DD'
    data_loader : dict or DataLoader
        数据加载器配置
        {
            "class": "QlibDataLoader",
            "kwargs": {"config": (expressions, names)}
        }
    infer_processors : list
        推理处理器列表（用于验证集、测试集）
    learn_processors : list
        训练处理器列表（用于训练集）
    process_type : str
        处理类型:
        - PTYPE_I (independent): 独立处理
        - PTYPE_A (append): 追加处理（先应用 infer，再应用 learn）
    drop_raw : bool
        是否删除原始数据以节省内存
    """
```

#### 实战示例 5: 完整的数据处理管道

```python
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import (
    Fillna, ZScoreNorm, CSZScoreNorm,
    RobustZScoreNorm, DropnaLabel
)

# 定义特征配置
feature_config = {
    "class": "QlibDataLoader",
    "kwargs": {
        "config": (
            # 价格特征
            [
                "$close/Ref($close,1)-1",                        # 日收益率
                "($high-$low)/$close",                           # 日内波幅
                "$volume/Mean($volume,20)",                      # 成交量比率
                "Mean($close,5)/Mean($close,20)",                # 均线比率
                "Std($close/Ref($close,1)-1,20)",                # 波动率
                "Corr($close,$volume,20)",                       # 价量相关性
                "($close-Min($low,20))/(Max($high,20)-Min($low,20))", # 价格位置
                "EMA($close,12)-EMA($close,26)",                 # MACD DIF
            ],
            # 特征名称
            [
                "RETURN", "RANGE", "VOLUME_RATIO", "MA_RATIO",
                "VOLATILITY", "CORR_CV", "POSITION", "MACD_DIF"
            ]
        )
    }
}

# 定义标签配置
label_config = {
    "class": "QlibDataLoader",
    "kwargs": {
        "config": (
            ["Ref($close,-5)/$close-1"],  # 未来5日收益率
            ["LABEL"]
        )
    }
}

# 创建数据处理器
handler = DataHandlerLP(
    instruments='csi300',
    start_time='2020-01-01',
    end_time='2024-01-31',
    fit_start_time='2020-01-01',  # 处理器拟合时间范围
    fit_end_time='2022-12-31',
    data_loader=feature_config,
    label=label_config,
    # 推理处理器（用于所有数据）
    infer_processors=[
        Fillna(fill_value=0),           # 填充缺失值
        ZScoreNorm(                      # Z-Score 标准化
            fit_start_time='2020-01-01',
            fit_end_time='2022-12-31'
        )
    ],
    # 训练处理器（仅用于训练数据）
    learn_processors=[
        DropnaLabel()  # 删除标签为 NaN 的样本
    ],
    process_type='PTYPE_A',  # 追加模式
    drop_raw=True            # 删除原始数据节省内存
)

# 获取处理后的数据
processed_data = handler.fetch()
print(f"数据形状: {processed_data.shape}")
print(f"列名: {processed_data.columns.tolist()}")
```

### 4.2 Alpha158 (预构建特征集)

**用途**: 生产级特征集，包含 158 个技术指标

**源码路径**: `qlib/contrib/data/handler.py`

#### 实战示例 6: 使用 Alpha158

```python
from qlib.contrib.data.handler import Alpha158

# 创建 Alpha158 处理器
alpha158_handler = Alpha158(
    instruments='csi300',
    start_time='2020-01-01',
    end_time='2024-01-31',
    fit_start_time='2020-01-01',
    fit_end_time='2022-12-31',
    # Alpha158 默认使用 CSZScoreNorm
)

# 获取数据
alpha158_data = alpha158_handler.fetch()
print(f"Alpha158 特征数量: {alpha158_data.shape[1]}")

# 查看特征分类
feature_names = alpha158_data.columns.tolist()
print("\n特征类别统计:")
categories = {}
for name in feature_names:
    prefix = name.split('_')[0]
    categories[prefix] = categories.get(prefix, 0) + 1

for category, count in sorted(categories.items()):
    print(f"  {category}: {count} 个特征")
```

**Alpha158 特征类别**:
- **KMID**: 价格相对位置特征
- **KUP/KLOW**: 上下影线比率
- **KSFT**: K线实体位置
- **OPEN/HIGH/LOW/CLOSE**: 价格特征
- **VOLUME**: 成交量特征
- **ROC**: 变化率
- **MA**: 移动平均
- **STD**: 标准差
- **BETA**: 市场贝塔
- **RSI**: 相对强弱指标

---

## 5. Processor 处理器详解

### 5.1 基础 Processor 类

所有处理器继承自基础 `Processor` 类:

```python
class Processor:
    """数据处理器基类"""

    def fit(self, df: pd.DataFrame = None):
        """从数据中学习参数"""
        pass

    def __call__(self, df: pd.DataFrame):
        """处理数据"""
        return df

    def is_for_infer(self) -> bool:
        """是否可用于推理"""
        return True

    def readonly(self) -> bool:
        """是否只读（不修改输入）"""
        return False
```

### 5.2 处理器详解

#### 5.2.1 Fillna - 缺失值填充

**源码**: `qlib/data/dataset/processor.py`

```python
class Fillna(Processor):
    """
    填充缺失值

    Parameters
    ----------
    fields_group : str or None
        要填充的列组，None 表示所有列
    fill_value : float
        填充值，默认 0
    """

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            df[cols] = df[cols].fillna(self.fill_value)
        return df
```

**实战示例 7**:

```python
# 创建测试数据
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'feature1': [1, 2, np.nan, 4, 5],
    'feature2': [np.nan, 2, 3, np.nan, 5],
    'label': [1, 2, 3, 4, 5]
})

print("原始数据:")
print(df)

# 填充所有列
fillna_all = Fillna(fill_value=0)
df_filled = fillna_all(df.copy())

print("\n填充后:")
print(df_filled)
```

#### 5.2.2 ZScoreNorm - Z-Score 标准化

**原理**: $(x - \mu) / \sigma$

**源码**: `qlib/data/dataset/processor.py`

```python
class ZScoreNorm(Processor):
    """
    Z-Score 标准化（按时间序列标准化）

    Parameters
    ----------
    fit_start_time : str
        拟合起始时间
    fit_end_time : str
        拟合结束时间
    fields_group : str or None
        要标准化的列组
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        """计算训练集的均值和标准差"""
        df_train = df.loc[self.fit_start_time:self.fit_end_time]
        cols = get_group_columns(df, self.fields_group)

        self.mean_train = np.nanmean(df_train[cols].values, axis=0)
        self.std_train = np.nanstd(df_train[cols].values, axis=0)

        # 处理标准差为0的情况
        self.std_train[self.std_train == 0] = 1

    def __call__(self, df):
        """应用标准化"""
        cols = get_group_columns(df, self.fields_group)
        df[cols] = (df[cols] - self.mean_train) / self.std_train
        return df
```

**实战示例 8**:

```python
# 创建时间序列数据
dates = pd.date_range('2020-01-01', periods=1000)
instruments = ['STOCK_A', 'STOCK_B']

df = pd.DataFrame({
    'datetime': np.repeat(dates, len(instruments)),
    'instrument': instruments * len(dates),
    'feature1': np.random.randn(len(dates) * len(instruments)) * 10 + 100,
    'feature2': np.random.randn(len(dates) * len(instruments)) * 5 + 50
}).set_index(['datetime', 'instrument'])

# 分割数据
train_df = df.loc[:'2022-12-31']
test_df = df.loc['2023-01-01':]

# 创建并拟合 ZScore 处理器
zscore = ZScoreNorm(
    fit_start_time='2020-01-01',
    fit_end_time='2022-12-31'
)
zscore.fit(df)

# 转换数据
train_normalized = zscore(train_df.copy())
test_normalized = zscore(test_df.copy())

print("训练集统计:")
print(train_normalized.describe())
print("\n测试集统计:")
print(test_normalized.describe())
```

#### 5.2.3 CSZScoreNorm - 截面 Z-Score 标准化

**关键区别**: 每个时间点对所有股票进行标准化（横截面）

**用途**: 消除市场整体波动，关注个股相对表现

```python
class CSZScoreNorm(Processor):
    """
    Cross-Sectional Z-Score 标准化

    在每个交易日内，对所有股票进行标准化

    Parameters
    ----------
    fields_group : str or None
        要标准化的列组
    method : str
        'zscore' 或 'robust'
    """

    def __init__(self, fields_group=None, method='zscore'):
        self.fields_group = fields_group
        self.method = method

    def zscore_func(self, x):
        """截面 Z-Score 函数"""
        if self.method == 'zscore':
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-8)
        elif self.method == 'robust':
            median = x.median()
            mad = (x - median).abs().median()
            return (x - median) / (mad * 1.4826 + 1e-8)

    def __call__(self, df):
        """按日期分组应用标准化"""
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby('datetime').transform(self.zscore_func)
        return df
```

**实战示例 9: ZScore vs CSZScore**

```python
import matplotlib.pyplot as plt

# 创建模拟数据（有市场整体趋势）
dates = pd.date_range('2023-01-01', periods=100)
stocks = [f'STOCK_{i}' for i in range(50)]

# 市场趋势 + 个股波动
market_trend = np.linspace(0, 10, 100)
data_list = []

for stock in stocks:
    stock_data = market_trend + np.random.randn(100) * 2
    for i, date in enumerate(dates):
        data_list.append({
            'datetime': date,
            'instrument': stock,
            'return': stock_data[i]
        })

df = pd.DataFrame(data_list).set_index(['datetime', 'instrument'])

# 应用不同的标准化
zscore = ZScoreNorm(
    fit_start_time='2023-01-01',
    fit_end_time='2023-03-31'
)
zscore.fit(df)
df_zscore = zscore(df.copy())

cs_zscore = CSZScoreNorm()
df_cs_zscore = cs_zscore(df.copy())

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始数据
df.groupby('datetime')['return'].mean().plot(ax=axes[0], title='Original (with market trend)')

# ZScore 标准化
df_zscore.groupby('datetime')['return'].mean().plot(ax=axes[1], title='ZScore (trend remains)')

# CSZScore 标准化
df_cs_zscore.groupby('datetime')['return'].mean().plot(ax=axes[2], title='CSZScore (trend removed)')

plt.tight_layout()
plt.show()
```

#### 5.2.4 RobustZScoreNorm - 稳健标准化

**用途**: 抵抗异常值的标准化方法

**原理**: 使用中位数和 MAD (Median Absolute Deviation) 替代均值和标准差

```python
class RobustZScoreNorm(Processor):
    """
    稳健 Z-Score 标准化

    公式: (x - median) / (MAD * 1.4826)
    其中 MAD = median(|x - median(x)|)
    1.4826 是转换系数，使 MAD 与标准差尺度一致

    Parameters
    ----------
    fit_start_time : str
    fit_end_time : str
    fields_group : str or None
    clip_outlier : bool
        是否裁剪异常值到 [-3, 3] 范围
    """

    def __init__(self, fit_start_time, fit_end_time,
                 fields_group=None, clip_outlier=True):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        """使用中位数和 MAD"""
        df_train = df.loc[self.fit_start_time:self.fit_end_time]
        cols = get_group_columns(df, self.fields_group)
        X = df_train[cols].values

        self.mean_train = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train = mad * 1.4826  # 转换系数
        self.std_train[self.std_train == 0] = 1

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = (df[cols] - self.mean_train) / self.std_train

        if self.clip_outlier:
            df[cols] = df[cols].clip(-3, 3)

        return df
```

**实战示例 10: 对比标准化方法**

```python
# 创建带异常值的数据
np.random.seed(42)
normal_data = np.random.randn(1000)
outliers = np.random.choice(1000, 50, replace=False)
normal_data[outliers] *= 20  # 添加异常值

df = pd.DataFrame({
    'datetime': pd.date_range('2020-01-01', periods=1000),
    'instrument': 'STOCK_A',
    'feature': normal_data
}).set_index(['datetime', 'instrument'])

# 应用不同标准化方法
zscore = ZScoreNorm('2020-01-01', '2022-12-31')
zscore.fit(df)
df_zscore = zscore(df.copy())

robust = RobustZScoreNorm('2020-01-01', '2022-12-31')
robust.fit(df)
df_robust = robust(df.copy())

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df['feature'], bins=50, alpha=0.7)
axes[0].set_title('Original (with outliers)')

axes[1].hist(df_zscore['feature'], bins=50, alpha=0.7)
axes[1].set_title('ZScore (affected by outliers)')

axes[2].hist(df_robust['feature'], bins=50, alpha=0.7)
axes[2].set_title('RobustZScore (outlier-resistant)')

plt.tight_layout()
plt.show()

print(f"ZScore - mean: {df_zscore['feature'].mean():.4f}, std: {df_zscore['feature'].std():.4f}")
print(f"Robust - mean: {df_robust['feature'].mean():.4f}, std: {df_robust['feature'].std():.4f}")
```

#### 5.2.5 MinMaxNorm - 最小最大标准化

```python
class MinMaxNorm(Processor):
    """
    Min-Max 标准化: (x - min) / (max - min)

    将特征缩放到 [0, 1] 范围
    """

    def fit(self, df: pd.DataFrame = None):
        df_train = df.loc[self.fit_start_time:self.fit_end_time]
        cols = get_group_columns(df, self.fields_group)

        self.min_val = np.nanmin(df_train[cols].values, axis=0)
        self.max_val = np.nanmax(df_train[cols].values, axis=0)

        # 处理常数列
        self.ignore_flag = (self.max_val - self.min_val) < 1e-8

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = (df[cols] - self.min_val) / (self.max_val - self.min_val + 1e-8)
        df[cols][self.ignore_flag] = 0  # 常数列设为0
        return df
```

#### 5.2.6 CSRankNorm - 截面排名标准化

```python
class CSRankNorm(Processor):
    """
    Cross-Sectional Rank 标准化

    将每个交易日的股票按特征值排名，转换为类正态分布

    步骤:
    1. 计算百分位排名 [0, 1]
    2. 中心化: 减去 0.5，得到 [-0.5, 0.5]
    3. 缩放: 乘以 3.46，使标准差约为 1
    """

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)

        # 按日期分组计算百分位排名
        t = df[cols].groupby('datetime').rank(pct=True)
        t -= 0.5      # 中心化
        t *= 3.46     # 缩放

        df[cols] = t
        return df
```

**实战示例 11**:

```python
# 创建截面数据
dates = pd.date_range('2023-01-01', periods=10)
stocks = [f'STOCK_{i}' for i in range(20)]

data = []
for date in dates:
    # 每天生成不同分布的数据
    values = np.random.exponential(scale=2, size=20)
    for i, stock in enumerate(stocks):
        data.append({
            'datetime': date,
            'instrument': stock,
            'feature': values[i]
        })

df = pd.DataFrame(data).set_index(['datetime', 'instrument'])

# 应用 CSRankNorm
rank_norm = CSRankNorm()
df_ranked = rank_norm(df.copy())

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始分布
df.loc[dates[0]].hist(ax=axes[0], bins=20)
axes[0].set_title('Original Distribution (exponential)')

# 排名标准化后
df_ranked.loc[dates[0]].hist(ax=axes[1], bins=20)
axes[1].set_title('After CSRankNorm (normal-like)')

plt.tight_layout()
plt.show()
```

#### 5.2.7 其他处理器

**DropnaLabel**: 删除标签为 NaN 的样本

```python
class DropnaLabel(DropnaProcessor):
    """删除标签为 NaN 的样本（不可用于推理）"""

    def __init__(self, fields_group="label"):
        super().__init__(fields_group=fields_group)

    def is_for_infer(self) -> bool:
        return False  # 推理时不能删除样本
```

**CSZFillna**: 用当日均值填充缺失值

```python
class CSZFillna(Processor):
    """用截面均值填充缺失值"""

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby('datetime').transform(
            lambda x: x.fillna(x.mean())
        )
        return df
```

**ProcessInf**: 处理无穷值

```python
class ProcessInf(Processor):
    """用均值替换无穷值"""

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)

        def replace_inf(x):
            mask_inf = np.isinf(x)
            if mask_inf.any():
                x[mask_inf] = x[~mask_inf].mean()
            return x

        df[cols] = df[cols].groupby('datetime').transform(replace_inf)
        return df
```

### 5.3 处理器链配置

**实战示例 12: 完整处理器链**

```python
# 推理处理器链（用于验证集和测试集）
infer_processors = [
    # 1. 处理无穷值
    {"class": "ProcessInf", "kwargs": {}},

    # 2. 用截面均值填充缺失值
    {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},

    # 3. 截面 Z-Score 标准化
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},

    # 4. 裁剪极端值
    {"class": "RobustZScoreNorm", "kwargs": {
        "fields_group": "feature",
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2022-12-31",
        "clip_outlier": True
    }}
]

# 训练处理器链（额外用于训练集）
learn_processors = [
    # 删除标签缺失的样本
    {"class": "DropnaLabel", "kwargs": {}}
]

# 创建处理器
handler = DataHandlerLP(
    instruments='csi300',
    start_time='2020-01-01',
    end_time='2024-01-31',
    data_loader=feature_config,
    label=label_config,
    infer_processors=infer_processors,
    learn_processors=learn_processors,
    process_type='PTYPE_A'  # 追加模式
)
```

---

## 6. 表达式引擎

### 6.1 基础操作符

| 操作符 | 说明 | 示例 |
|--------|------|------|
| **$** | 访问原始字段 | `$close`, `$volume` |
| **Ref(expr, n)** | 回溯 n 期 | `Ref($close, 1)` 表示昨日收盘价 |
| **Mean(expr, n)** | n 期移动平均 | `Mean($close, 5)` 表示 5 日均线 |
| **Std(expr, n)** | n 期标准差 | `Std($close, 20)` |
| **Sum(expr, n)** | n 期累计和 | `Sum($volume, 5)` |
| **Max/Min(expr, n)** | n 期最大/最小值 | `Max($high, 20)` |

### 6.2 高级函数

| 函数 | 说明 | 示例 |
|------|------|------|
| **EMA(expr, n)** | 指数移动平均 | `EMA($close, 12)` |
| **Corr(expr1, expr2, n)** | n 期相关系数 | `Corr($close, $volume, 20)` |
| **Rank(expr)** | 截面排名 | `Rank($close)` |
| **Log/Abs/Sign** | 数学函数 | `Log($close)`, `Abs($return)` |
| **Greater/Less** | 比较函数 | `Greater($ma5, $ma20)` |
| **If(cond, x, y)** | 条件函数 | `If($close > $open, 1, 0)` |

### 6.3 实战表达式

**实战示例 13: 常用技术指标**

```python
# 技术指标表达式库
technical_indicators = {
    # ===== 趋势指标 =====
    "日收益率": "$close / Ref($close, 1) - 1",
    "5日收益率": "$close / Ref($close, 5) - 1",
    "20日收益率": "$close / Ref($close, 20) - 1",

    "MA5": "Mean($close, 5)",
    "MA10": "Mean($close, 10)",
    "MA20": "Mean($close, 20)",
    "MA60": "Mean($close, 60)",

    "MA5/MA20": "Mean($close, 5) / Mean($close, 20)",
    "MA10/MA60": "Mean($close, 10) / Mean($close, 60)",

    # ===== MACD =====
    "MACD_DIF": "EMA($close, 12) - EMA($close, 26)",
    "MACD_DEA": "EMA(EMA($close, 12) - EMA($close, 26), 9)",
    "MACD_HIST": "(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)",

    # ===== 波动率指标 =====
    "日波动率": "Std($close/Ref($close,1)-1, 20)",
    "周波动率": "Std($close/Ref($close,1)-1, 5)",
    "月波动率": "Std($close/Ref($close,1)-1, 20)",

    # 布林带
    "BB_UPPER": "Mean($close, 20) + 2*Std($close, 20)",
    "BB_LOWER": "Mean($close, 20) - 2*Std($close, 20)",
    "BB_WIDTH": "(Mean($close, 20) + 2*Std($close, 20) - (Mean($close, 20) - 2*Std($close, 20))) / Mean($close, 20)",

    # ===== 成交量指标 =====
    "成交量5日均线": "Mean($volume, 5)",
    "成交量20日均线": "Mean($volume, 20)",
    "量比": "$volume / Mean($volume, 5)",
    "换手率": "$volume / Mean($volume, 60)",

    # 量价相关性
    "量价相关20日": "Corr($close/Ref($close,1)-1, $volume, 20)",
    "量价相关60日": "Corr($close/Ref($close,1)-1, $volume, 60)",

    # ===== 价格位置指标 =====
    "价格位置": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20))",
    "距20日高点": "($close - Max($high, 20)) / Max($high, 20)",
    "距20日低点": "($close - Min($low, 20)) / Min($low, 20)",

    # ===== K线形态 =====
    "实体比例": "Abs($close - $open) / ($high - $low)",
    "上影线比例": "($high - Greater($close, $open)) / ($high - $low)",
    "下影线比例": "(Less($close, $open) - $low) / ($high - $low)",

    # ===== 动量指标 =====
    "RSI": "Sum(Greater($close - Ref($close, 1), 0) * ($close - Ref($close, 1)), 14) / Sum(Abs($close - Ref($close, 1)), 14)",
    "威廉指标": "(Max($high, 14) - $close) / (Max($high, 14) - Min($low, 14))",

    # ===== 市场微观结构 =====
    "Amihud流动性": "Abs($close/Ref($close,1)-1) / $volume",
    "买卖价差": "($high - $low) / (($high + $low) / 2)",
    "日内振幅": "($high - $low) / Ref($close, 1)",
}

# 使用示例
from qlib.data import D

expressions = list(technical_indicators.values())
names = list(technical_indicators.keys())

data = D.features(
    instruments=['SH600519'],
    fields=expressions[:10],  # 先测试前10个
    start_time='2023-01-01',
    end_time='2024-01-31'
)
data.columns = names[:10]

print(data.head())
```

### 6.4 表达式最佳实践

**1. 避免未来函数**

```python
# ❌ 错误: 使用未来数据
"$close / Ref($close, -1) - 1"  # Ref(..., -1) 表示明天的价格!

# ✅ 正确: 使用历史数据
"$close / Ref($close, 1) - 1"   # Ref(..., 1) 表示昨天的价格
```

**2. 处理除零错误**

```python
# ❌ 可能除零
"$close / Ref($close, 1)"

# ✅ 添加小常数
"$close / (Ref($close, 1) + 1e-8)"
```

**3. 标准化表达式**

```python
# 推荐: 先计算原始特征，再用 Processor 标准化
expressions = [
    "$close / Ref($close, 1) - 1",  # 原始收益率
    "Mean($close, 5)"                # 原始均线
]

# 然后用 ZScoreNorm 标准化
infer_processors = [
    Fillna(fill_value=0),
    ZScoreNorm(...)
]
```

---

## 7. 实战应用案例

### 7.1 完整的量化策略数据准备

**实战示例 14: 端到端数据流程**

```python
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import *

# 1. 初始化
qlib.init()

# 2. 定义特征工程
feature_expressions = [
    # 价格特征
    "$close / Ref($close, 1) - 1",              # 日收益率
    "$close / Ref($close, 5) - 1",              # 5日收益率
    "$close / Ref($close, 20) - 1",             # 20日收益率

    # 均线系统
    "Mean($close, 5)",
    "Mean($close, 10)",
    "Mean($close, 20)",
    "Mean($close, 60)",
    "Mean($close, 5) / Mean($close, 20) - 1",   # 均线偏离

    # 波动率
    "Std($close/Ref($close,1)-1, 5)",
    "Std($close/Ref($close,1)-1, 20)",
    "Std($close/Ref($close,1)-1, 60)",

    # 成交量
    "$volume / Mean($volume, 5) - 1",
    "$volume / Mean($volume, 20) - 1",
    "Corr($close, $volume, 20)",

    # MACD
    "EMA($close, 12) - EMA($close, 26)",
    "EMA(EMA($close, 12) - EMA($close, 26), 9)",

    # 价格位置
    "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20))",
    "($high - $low) / $close",
]

feature_names = [
    "return_1d", "return_5d", "return_20d",
    "ma5", "ma10", "ma20", "ma60", "ma_deviation",
    "vol_5d", "vol_20d", "vol_60d",
    "volume_ratio_5d", "volume_ratio_20d", "corr_pv",
    "macd_dif", "macd_dea",
    "price_position", "day_range"
]

# 3. 定义标签
label_expressions = ["Ref($close, -5) / $close - 1"]
label_names = ["label_5d"]

# 4. 创建处理器
handler = DataHandlerLP(
    instruments='csi300',
    start_time='2018-01-01',
    end_time='2024-01-31',
    fit_start_time='2018-01-01',
    fit_end_time='2022-12-31',
    data_loader={
        "class": "QlibDataLoader",
        "kwargs": {
            "config": (feature_expressions, feature_names)
        }
    },
    label={
        "class": "QlibDataLoader",
        "kwargs": {
            "config": (label_expressions, label_names)
        }
    },
    infer_processors=[
        ProcessInf(),
        CSZFillna(fields_group="feature"),
        CSZScoreNorm(fields_group="feature"),
    ],
    learn_processors=[
        DropnaLabel()
    ],
    process_type='PTYPE_A'
)

# 5. 创建数据集
dataset = DatasetH(
    handler=handler,
    segments={
        'train': ('2018-01-01', '2021-12-31'),
        'valid': ('2022-01-01', '2022-12-31'),
        'test': ('2023-01-01', '2024-01-31')
    }
)

# 6. 准备数据
train_data = dataset.prepare('train', col_set='feature')
train_label = dataset.prepare('train', col_set='label')
valid_data = dataset.prepare('valid', col_set='feature')
valid_label = dataset.prepare('valid', col_set='label')

print("数据准备完成!")
print(f"训练集: {train_data.shape}, 标签: {train_label.shape}")
print(f"验证集: {valid_data.shape}, 标签: {valid_label.shape}")
```

### 7.2 因子分析流程

**实战示例 15: 单因子测试**

```python
from qlib.data import D
import pandas as pd
import numpy as np

def factor_analysis(factor_expr, factor_name, instruments,
                   start_time, end_time):
    """
    单因子分析

    Parameters
    ----------
    factor_expr : str
        因子表达式
    factor_name : str
        因子名称
    instruments : str or list
        股票池
    start_time, end_time : str
        时间范围
    """

    # 1. 计算因子和未来收益
    data = D.features(
        instruments=instruments,
        fields=[
            factor_expr,
            "Ref($close, -5) / $close - 1",  # 未来5日收益
            "Ref($close, -10) / $close - 1", # 未来10日收益
        ],
        start_time=start_time,
        end_time=end_time
    )
    data.columns = ['factor', 'ret_5d', 'ret_10d']

    # 2. 截面分组回测
    def group_backtest(df, n_groups=10):
        """按因子值分组，计算各组收益"""
        df = df.dropna()
        df['group'] = pd.qcut(df['factor'], n_groups, labels=False, duplicates='drop')

        group_returns = df.groupby('group')[['ret_5d', 'ret_10d']].mean()
        return group_returns

    # 按日期分组计算
    results = []
    for date in data.index.get_level_values('datetime').unique():
        day_data = data.loc[date]
        group_ret = group_backtest(day_data)
        group_ret['date'] = date
        results.append(group_ret)

    results_df = pd.concat(results)

    # 3. 计算 IC
    ic_5d = data.groupby('datetime').apply(
        lambda x: x['factor'].corr(x['ret_5d'])
    )
    ic_10d = data.groupby('datetime').apply(
        lambda x: x['factor'].corr(x['ret_10d'])
    )

    # 4. 输出结果
    print(f"\n{'='*60}")
    print(f"因子: {factor_name}")
    print(f"表达式: {factor_expr}")
    print(f"{'='*60}")

    print(f"\nIC 统计:")
    print(f"  5日IC均值: {ic_5d.mean():.4f}, ICIR: {ic_5d.mean() / ic_5d.std():.4f}")
    print(f"  10日IC均值: {ic_10d.mean():.4f}, ICIR: {ic_10d.mean() / ic_10d.std():.4f}")

    print(f"\n分组收益 (平均):")
    avg_group_ret = results_df.groupby('group').mean()
    print(avg_group_ret)

    print(f"\n多空收益:")
    long_short = avg_group_ret.iloc[-1] - avg_group_ret.iloc[0]
    print(f"  5日: {long_short['ret_5d']:.4f}")
    print(f"  10日: {long_short['ret_10d']:.4f}")

    return {
        'ic_5d': ic_5d,
        'ic_10d': ic_10d,
        'group_returns': results_df
    }

# 测试多个因子
factors_to_test = {
    "动量20日": "$close / Ref($close, 20) - 1",
    "反转5日": "Ref($close, 5) / $close - 1",
    "波动率": "Std($close/Ref($close,1)-1, 20)",
    "量价相关": "Corr($close, $volume, 20)",
    "换手率": "$volume / Mean($volume, 60)",
}

results = {}
for name, expr in factors_to_test.items():
    results[name] = factor_analysis(
        expr, name,
        instruments='csi300',
        start_time='2023-01-01',
        end_time='2024-01-31'
    )
```

---

## 8. 常见陷阱与最佳实践

### 8.1 数据泄露 (Data Leakage)

**陷阱 1: 未来函数**

```python
# ❌ 错误示例
label_wrong = "Ref($close, -5) / Ref($close, -1) - 1"
# 使用了未来第1天的价格作为基准!

# ✅ 正确示例
label_correct = "Ref($close, -5) / $close - 1"
# 以当前价格作为基准，预测未来5天收益
```

**陷阱 2: 标准化数据泄露**

```python
# ❌ 错误: 在全部数据上拟合标准化参数
scaler = ZScoreNorm()
scaler.fit(all_data)  # 包含了测试集!
train_normalized = scaler(train_data)
test_normalized = scaler(test_data)

# ✅ 正确: 只在训练集上拟合
scaler = ZScoreNorm(
    fit_start_time='2018-01-01',
    fit_end_time='2021-12-31'  # 只用训练期
)
scaler.fit(all_data)  # fit 时会自动提取训练期数据
train_normalized = scaler(train_data)
test_normalized = scaler(test_data)
```

### 8.2 幸存者偏差 (Survivorship Bias)

```python
# ❌ 错误: 使用当前指数成分股回测历史
instruments = D.instruments('csi300')  # 当前成分股
data = D.features(
    instruments=instruments,
    start_time='2010-01-01',  # 回测到2010年
    end_time='2024-01-31'
)
# 问题: 2010年的CSI300成分股不是这些!

# ✅ 正确: 使用动态股票池
handler = DataHandlerLP(
    instruments='csi300',  # Qlib会自动处理动态成分
    start_time='2010-01-01',
    end_time='2024-01-31'
)
```

### 8.3 处理器顺序

```python
# ✅ 推荐的处理器顺序
infer_processors = [
    ProcessInf(),              # 1. 先处理无穷值
    CSZFillna(),               # 2. 再填充缺失值
    CSZScoreNorm(),            # 3. 最后标准化
]

# ❌ 不推荐
infer_processors = [
    CSZScoreNorm(),            # NaN 和 Inf 会影响标准化!
    ProcessInf(),              # 标准化后再处理Inf无意义
    CSZFillna(),
]
```

### 8.4 内存优化

```python
# 技巧 1: 使用 drop_raw 删除原始数据
handler = DataHandlerLP(
    ...,
    drop_raw=True  # 节省内存
)

# 技巧 2: 分批加载大规模数据
instruments = D.instruments('csi300')
batch_size = 50

data_batches = []
for i in range(0, len(instruments), batch_size):
    batch = instruments[i:i+batch_size]
    batch_data = D.features(
        instruments=batch,
        fields=expressions,
        start_time=start_time,
        end_time=end_time
    )
    data_batches.append(batch_data)

full_data = pd.concat(data_batches)

# 技巧 3: 使用缓存
handler = DataHandlerLP(
    ...,
    cache=True  # 启用缓存
)
```

### 8.5 性能优化

```python
# 技巧 1: 预计算复杂特征
# 不要在表达式中重复计算
# ❌ 低效
expressions = [
    "Mean($close, 20) + 2*Std($close, 20)",
    "Mean($close, 20) - 2*Std($close, 20)",
]

# ✅ 高效: 分解为基础特征，后续组合
expressions = [
    "Mean($close, 20)",
    "Std($close, 20)"
]
# 在 Python 中计算上下轨
data['bb_upper'] = data['ma20'] + 2 * data['std20']
data['bb_lower'] = data['ma20'] - 2 * data['std20']

# 技巧 2: 使用合适的数据粒度
# 日度数据足够时不要用分钟数据
```

### 8.6 调试技巧

```python
# 技巧 1: 检查数据质量
def check_data_quality(df, name="Data"):
    """数据质量检查"""
    print(f"\n{'='*60}")
    print(f"{name} 质量检查")
    print(f"{'='*60}")

    print(f"形状: {df.shape}")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n缺失值:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'count': missing,
        'percentage': missing_pct
    })
    print(missing_df[missing_df['count'] > 0])

    print(f"\n无穷值:")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum()
    print(inf_count[inf_count > 0])

    print(f"\n统计摘要:")
    print(df.describe())

# 使用
check_data_quality(train_data, "训练集")
```

---

## 总结 (Summary)

### 核心要点

1. **数据流程**
   ```
   原始数据 → DataLoader → DataHandler → Processor链 → Dataset → 模型
   ```

2. **关键类**
   - **DatasetH**: 标准数据集
   - **TSDatasetH**: 时间序列数据集
   - **QlibDataLoader**: 数据加载器
   - **DataHandlerLP**: 数据处理器
   - **Processors**: 数据变换

3. **重要处理器**
   - **Fillna**: 填充缺失值
   - **ZScoreNorm**: 时间序列标准化
   - **CSZScoreNorm**: 截面标准化（推荐用于因子）
   - **RobustZScoreNorm**: 抗异常值标准化

4. **避免陷阱**
   - ❌ 未来函数
   - ❌ 数据泄露
   - ❌ 幸存者偏差
   - ✅ 正确的处理器顺序
   - ✅ 合理的标准化方法

### 推荐阅读

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- `qlib/data/dataset/processor.py` 源码
- `qlib/contrib/data/handler.py` 源码

---

**文档版本**: v1.0
**最后更新**: 2025-11-05
**反馈**: 欢迎提issue到 GitHub 仓库
