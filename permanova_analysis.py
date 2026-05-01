"""
PERMANOVA 分析：检验各水动力变量对过鱼成功率(PSR)的影响
==========================================================

方法说明
--------
PERMANOVA (Permutational Multivariate Analysis of Variance) 是多元方差分析的非参数版本。
它不要求数据满足正态分布假设，而是通过置换检验来评估统计显著性。

核心思想：
  1. 用距离矩阵描述样本间的差异（本文使用欧氏距离）
  2. 将总变异分解为"由解释变量引起的变异"和"残差变异"
  3. 通过随机置换来构建 F 统计量的零分布，从而计算 p 值

关键参数解释
-----------
  F 统计量：组间变异 / 组内变异 的比值。F 越大，说明解释变量对响应变量的影响越强。
  R² (决定系数)：解释变量所能解释的总变异比例。R² = SS_A / SS_T，取值 0~1。
      - R² = 0.88 表示该变量解释了 88% 的 PSR 变异
  p 值：在"解释变量无影响"的零假设下，观测到当前或更极端 F 值的概率。
      - p < 0.05 通常认为显著
      - p < 0.001 认为极显著

数学过程
--------
  1. 构建响应变量(PSR)的欧氏距离矩阵 D，其中 D_ij = |PSR_i - PSR_j|
  2. 对 D 做 Gower 中心化：A = -0.5 * D², 然后 G = A - 行均值 - 列均值 + 总均值
     G 矩阵的对角线之和(trace)即为总平方和 SS_T
  3. 构建解释变量的投影矩阵(Hat matrix)：H = X(X'X)⁻¹X'
     其中 X = [1, x] 包含截距项和解释变量
  4. 模型平方和 SS_A = trace(H G H)，残差平方和 SS_R = SS_T - SS_A
  5. F = (SS_A / df_A) / (SS_R / df_R)
  6. 置换检验：随机打乱解释变量 9999 次，每次重新计算 F，统计有多少次 F_perm >= F_obs
     p = (count + 1) / (n_perm + 1)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# 核心函数：单变量 PERMANOVA
# ---------------------------------------------------------------------------
def permanova_single(y, x, n_perm=9999, seed=42):
    """
    单变量 PERMANOVA：检验一个连续解释变量 x 对响应变量 y 的影响。

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        响应变量（如 PSR）。
    x : np.ndarray, shape (n,)
        解释变量（如某个水动力指标）。
    n_perm : int
        置换次数，默认 9999。次数越多，p 值估计越稳定。
    seed : int
        随机种子，保证结果可复现。

    Returns
    -------
    F_obs : float
        观测到的 F 统计量。
    R2 : float
        决定系数，解释变量对总变异的解释比例。
    p_value : float
        置换检验 p 值。
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # --- Step 1: 构建欧氏距离矩阵并做 Gower 中心化 ---
    D = squareform(pdist(y.reshape(-1, 1), metric="euclidean"))
    A = -0.5 * D ** 2
    G = A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()

    # 总平方和
    SS_T = np.trace(G)

    # --- Step 2: 构建 Hat 矩阵，计算模型平方和 ---
    X = np.column_stack([np.ones(n), x])          # 设计矩阵 [截距, x]
    H = X @ np.linalg.pinv(X.T @ X) @ X.T         # 投影矩阵
    SS_A = np.trace(H @ G @ H)                     # 模型平方和
    SS_R = SS_T - SS_A                              # 残差平方和

    # 自由度
    df_A = 1                                        # 1 个解释变量
    df_R = n - 2                                    # n - (截距 + 1个变量)

    # F 统计量与 R²
    F_obs = (SS_A / df_A) / (SS_R / df_R)
    R2 = SS_A / SS_T

    # --- Step 3: 置换检验 ---
    count = 0
    for _ in range(n_perm):
        x_perm = x[rng.permutation(n)]
        X_perm = np.column_stack([np.ones(n), x_perm])
        H_perm = X_perm @ np.linalg.pinv(X_perm.T @ X_perm) @ X_perm.T
        SS_A_perm = np.trace(H_perm @ G @ H_perm)
        SS_R_perm = SS_T - SS_A_perm
        F_perm = (SS_A_perm / df_A) / (SS_R_perm / df_R)
        if F_perm >= F_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return F_obs, R2, p_value


# ---------------------------------------------------------------------------
# 核心函数：多变量联合 PERMANOVA
# ---------------------------------------------------------------------------
def permanova_multi(y, X_vars, n_perm=9999, seed=42):
    """
    多变量联合 PERMANOVA：检验多个解释变量共同对响应变量 y 的影响。

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        响应变量。
    X_vars : np.ndarray, shape (n, p)
        多个解释变量组成的矩阵。
    n_perm : int
        置换次数。
    seed : int
        随机种子。

    Returns
    -------
    F_obs, R2, p_value : float
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    p = X_vars.shape[1]

    D = squareform(pdist(y.reshape(-1, 1), metric="euclidean"))
    A = -0.5 * D ** 2
    G = A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()
    SS_T = np.trace(G)

    X = np.column_stack([np.ones(n), X_vars])
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    SS_A = np.trace(H @ G @ H)
    SS_R = SS_T - SS_A

    df_A = p
    df_R = n - p - 1
    F_obs = (SS_A / df_A) / (SS_R / df_R)
    R2 = SS_A / SS_T

    count = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        X_perm = np.column_stack([np.ones(n), X_vars[perm_idx]])
        H_perm = X_perm @ np.linalg.pinv(X_perm.T @ X_perm) @ X_perm.T
        SS_A_perm = np.trace(H_perm @ G @ H_perm)
        SS_R_perm = SS_T - SS_A_perm
        F_perm = (SS_A_perm / df_A) / (SS_R_perm / df_R)
        if F_perm >= F_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return F_obs, R2, p_value


# ---------------------------------------------------------------------------
# 辅助函数：对单个响应变量运行完整分析并打印结果
# ---------------------------------------------------------------------------
def run_analysis(df, response_col, response_name, hydro_vars):
    """
    对指定响应变量，逐一检验每个水动力变量的影响，并做多变量联合检验。

    返回结果列表，用于后续导出 CSV。
    """
    y = df[response_col].values.astype(float)
    rows = []

    # 单变量检验
    print("=" * 72)
    print(f"PERMANOVA 单变量检验：各水动力变量对 {response_name}({response_col}) 的影响")
    print("距离度量: 欧氏距离 | 置换次数: 9999")
    print("=" * 72)
    print(f"{'变量':<6} {'名称':<14} {'F统计量':>10} {'R²':>10} {'p值':>10} {'显著性':>6}")
    print("-" * 72)

    for var, name in hydro_vars.items():
        x = df[var].values.astype(float)
        F, R2, p = permanova_single(y, x)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{var:<6} {name:<14} {F:>10.4f} {R2:>10.4f} {p:>10.4f} {sig:>6}")
        rows.append({
            "响应变量": response_col,
            "响应变量名称": response_name,
            "检验类型": "单变量",
            "解释变量": var,
            "解释变量名称": name,
            "F统计量": round(F, 4),
            "R²": round(R2, 4),
            "p值": round(p, 4),
            "显著性": sig,
        })

    print("-" * 72)
    print("显著性: *** p<0.001  ** p<0.01  * p<0.05  ns 不显著\n")

    # 多变量联合检验
    X_all = df[list(hydro_vars.keys())].values.astype(float)
    F_all, R2_all, p_all = permanova_multi(y, X_all)
    sig_all = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else "ns"

    print("=" * 72)
    print(f"PERMANOVA 多变量联合检验：所有水动力变量共同对 {response_name}({response_col}) 的影响")
    print("=" * 72)
    print(f"F统计量:       {F_all:.4f}")
    print(f"R² (解释度):   {R2_all:.4f} ({R2_all * 100:.1f}%)")
    print(f"p值:           {p_all:.4f}")
    print("\n")

    rows.append({
        "响应变量": response_col,
        "响应变量名称": response_name,
        "检验类型": "多变量联合",
        "解释变量": "Um+km+tm+Umx+kmx+tmx",
        "解释变量名称": "全部水动力变量",
        "F统计量": round(F_all, 4),
        "R²": round(R2_all, 4),
        "p值": round(p_all, 4),
        "显著性": sig_all,
    })

    return rows


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    # 读取数据
    df = pd.read_csv("Table3.csv")

    # 待检验的水动力变量及其中文名称
    hydro_vars = {
        "Um":  "平均流速",
        "km":  "平均湍流动能",
        "tm":  "紊动强度",
        "Umx": "最大流速",
        "kmx": "最大湍流动能",
        "tmx": "最大紊动强度",
    }

    # 4 个响应变量
    response_vars = {
        "PSR":  "过鱼成功率",
        "MFTT": "平均首次通过时间",
        "MET":  "平均探索时间",
        "EE":   "进入效率",
    }

    all_rows = []
    for col, name in response_vars.items():
        rows = run_analysis(df, col, name, hydro_vars)
        all_rows.extend(rows)

    # 导出 CSV
    result_df = pd.DataFrame(all_rows)
    output_path = "permanova_results.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
