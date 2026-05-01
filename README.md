# PERMANOVA Fish Passage Analysis

PERMANOVA（置换多元方差分析）检验各水动力变量对鱼道过鱼性能指标的影响。

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

将 `Table3.csv` 放置在项目根目录下（与 `permanova_analysis.py` 同级）：

```
permanova-fish-passage/
├── Table3.csv              <- 数据文件放这里
├── permanova_analysis.py
├── requirements.txt
└── README.md
```

`Table3.csv` 格式要求（CSV，含表头）：

| 列名 | 含义 |
|------|------|
| PSR | 过鱼成功率 |
| MFTT | 平均首次通过时间 |
| MET | 平均探索时间 |
| EE | 进入效率 |
| Um | 平均流速 |
| km | 平均湍流动能 |
| tm | 紊动强度 |
| Umx | 最大流速 |
| kmx | 最大湍流动能 |
| tmx | 最大紊动强度 |

## 运行

```bash
python permanova_analysis.py
```

运行后会在终端输出各变量的 F 统计量、R²、p 值，并将结果导出为 `permanova_results.csv`。
