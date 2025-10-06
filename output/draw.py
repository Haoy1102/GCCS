# plot_results.py
# Matplotlib only; one chart per figure.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== settings ==================
OUT_DIR = Path("./figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# —— 基准线配置（不做归一化）——
# 基准线两种模式： "method" 用某个方法的曲线作为参考；"constant" 用常数水平线
BASELINE_MODE     = "method"        # "method" 或 "constant"
BASELINE_METHOD   = "GCCS"          # 当 BASELINE_MODE="method" 时，参考的方法名
BASELINE_CONST_Y  = 300.0           # 当 BASELINE_MODE="constant" 时，水平线的 y 值（示例）
# === E1 图样式开关： "bar" 或 "line" ===
E1_STYLE = "bar"   # 想用折线就改成 "line"


from pathlib import Path

DATA_DIR = Path("data_gen")
# CSV_E1_EQUAL = DATA_DIR / "e1_equal.csv"
CSV_E1_EQUAL = "./data/result.csv"
CSV_E1_UNEQUAL = DATA_DIR / "e1_unequal.csv"
CSV_E2 = DATA_DIR / "e2_heterogeneity.csv"
CSV_E3 = DATA_DIR / "e3_balanced_longtail.csv"
CSV_E4 = DATA_DIR / "e4_ablation.csv"

# 方法重命名
METHOD_ORDER = ["GCCS", "HEFT", "Hydra", "MRSA"]

# 固定配色（参考你的图：红、蓝、橙、绿）
COLORS = {
    "GCCS": "#E41A1C",  # red
    "HEFT": "#377EB8",  # blue
    "Hydra": "#FF7F00", # orange
    "MRSA": "#4DAF4A",  # green
}

# —— 饱和度（透明度）开关：True=使用alpha，False=纯色 ——
USE_ALPHA = True
BAR_ALPHA  = 0.85
LINE_ALPHA = 0.90

# —— 白色间隙样式（边框为白色，略窄的柱宽） ——
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.6
GROUP_TOTAL_WIDTH = 0.78   # 每组总体宽度（<1 会让组之间也留些空隙）
# ==============================================


def _csv_exists():
    return all(Path(p).exists() for p in [CSV_E1_EQUAL, CSV_E1_UNEQUAL, CSV_E2, CSV_E3, CSV_E4])

def load_data():
    if not _csv_exists():
        missing = [p for p in [CSV_E1_EQUAL, CSV_E1_UNEQUAL, CSV_E2, CSV_E3, CSV_E4] if not Path(p).exists()]
        raise FileNotFoundError(f"缺少以下 CSV 文件：{missing}。请先放置数据再运行。")

    e1_equal = pd.read_csv(CSV_E1_EQUAL)
    e1_unequal = pd.read_csv(CSV_E1_UNEQUAL)
    e2 = pd.read_csv(CSV_E2)
    e3 = pd.read_csv(CSV_E3)
    e4 = pd.read_csv(CSV_E4)

    return e1_equal, e1_unequal, e2, e3, e4

def _ordered_methods(cands):
    cset = list(dict.fromkeys(cands))
    ordered = [m for m in METHOD_ORDER if m in cset]
    tail = [m for m in cset if m not in METHOD_ORDER]
    return ordered + tail

def grouped_bar(ax, x_labels, series_dict, title, xlabel, ylabel):
    methods = _ordered_methods(series_dict.keys())
    x = np.arange(len(x_labels), dtype=float)
    width = GROUP_TOTAL_WIDTH / max(1, len(methods))  # 让组内留白
    for idx, m in enumerate(methods):
        ax.bar(
            x + (idx - (len(methods)-1)/2)*width,
            series_dict[m],
            width * 0.94,  # 略缩柱宽，柱与柱之间留更细的缝
            label=m,
            color=COLORS.get(m, None),
            edgecolor=BAR_EDGE_COLOR,     # 白色细描边 → 淡淡白缝
            linewidth=BAR_EDGE_WIDTH,
            alpha=(BAR_ALPHA if USE_ALPHA else 1.0),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

def plot_e1_for_rho(df, rho, tag):
    df_r = df[df["rho"] == rho].copy()
    kappas = sorted(df_r["kappa"].unique())
    methods = _ordered_methods(sorted(df_r["method"].unique()))

    if E1_STYLE == "line":
        # —— 折线图 ——（点白边、y 轴虚线网格、图例带框）
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for m in methods:
            d = df_r[df_r["method"] == m].sort_values("kappa")
            ax.plot(
                d["kappa"].values,
                d["makespan"].values,
                marker="o",
                label=m,
                color=COLORS.get(m, None),
                linewidth=2,
                alpha=(LINE_ALPHA if USE_ALPHA else 1.0),
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        ax.set_xticks(kappas)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    else:
        # —— 柱状图 ——（留白缝、白描边，与你当前 grouped_bar 风格一致）
        series = {
            m: [df_r[(df_r["method"] == m) & (df_r["kappa"] == k)]["makespan"].values[0]
                for k in kappas]
            for m in methods
        }
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        grouped_bar(
            ax,
            [str(k) for k in kappas],
            series,
            title="",  # 标题下面统一设置
            xlabel="",
            ylabel="",
        )

    ax.set_title(f"E1-{tag}: makespan vs kappa (rho={rho})")
    ax.set_xlabel("kappa (vGPU count per server)")
    ax.set_ylabel("Makespan")

    # 图例带框
    leg = ax.legend(frameon=True)
    leg.get_frame().set_linewidth(0.8)

    out = OUT_DIR / f"fig_E1_{tag}_rho{str(rho).replace('.','_')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_e2(field, title, filename, df):
    fig, ax = plt.subplots(figsize=(7.6,4.4))
    for m in _ordered_methods(df["method"].unique()):
        d = df[df["method"]==m].sort_values("H")
        ax.plot(
            d["H"].values, d[field].values,
            marker="o",
            label=m,
            color=COLORS.get(m, None),
            linewidth=2,
            alpha=(LINE_ALPHA if USE_ALPHA else 1.0),
            markeredgecolor="white", markeredgewidth=0.6  # 折线的点也有细白边，风格统一
        )
    ax.set_title(title)
    ax.set_xlabel("H (CV of vGPU capacity split)")
    ax.set_ylabel(field.replace("_"," ").title())
    ax.legend(frameon=False)
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_e3(metric, title_suffix, filename, df):
    fig, ax = plt.subplots(figsize=(7.6,4.4))
    datasets = ["Balanced","LongTail"]
    methods = _ordered_methods(df["method"].unique())
    x = np.arange(len(datasets), dtype=float)
    width = GROUP_TOTAL_WIDTH / max(1, len(methods))
    for idx, m in enumerate(methods):
        vals = [df[(df["dataset"]==d) & (df["method"]==m)][metric].values[0] for d in datasets]
        ax.bar(
            x + (idx - (len(methods)-1)/2)*width,
            vals,
            width * 0.94,
            label=m,
            color=COLORS.get(m, None),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            alpha=(BAR_ALPHA if USE_ALPHA else 1.0),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_title(f"E3: {title_suffix}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.upper())
    ax.legend(frameon=False)
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_e4(metric, title_suffix, filename, df):
    fig, ax = plt.subplots(figsize=(7.6,4.4))
    x_labels = list(df["variant"].values)
    x = np.arange(len(x_labels))
    ax.bar(
        x, list(df[metric].values),
        color="#BDBDBD",
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        alpha=(BAR_ALPHA if USE_ALPHA else 1.0),
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=12, ha="right")
    ax.set_title(f"E4: {title_suffix}")
    ax.set_xlabel("Variant")
    ax.set_ylabel(metric.upper())
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def main():
    e1_equal, e1_unequal, e2, e3, e4 = load_data()

    for rho in sorted(e1_equal["rho"].unique()):
        plot_e1_for_rho(e1_equal, rho, "equal")
    for rho in sorted(e1_unequal["rho"].unique()):
        plot_e1_for_rho(e1_unequal, rho, "unequal")

    if {"H","method","makespan_multiplier"}.issubset(e2.columns):
        plot_e2("makespan_multiplier", "E2: makespan multiplier vs heterogeneity H", "fig_E2_makespan_multiplier.png", e2)
    # if {"H","method","p99_multiplier"}.issubset(e2.columns):
    #     plot_e2("p99_multiplier", "E2: P99 multiplier vs heterogeneity H", "fig_E2_p99_multiplier.png", e2)

    if {"dataset","method","makespan"}.issubset(e3.columns):
        plot_e3("makespan", "makespan on Balanced vs LongTail", "fig_E3_makespan.png", e3)
    # if {"dataset","method","p99"}.issubset(e3.columns):
    #     plot_e3("p99", "P99 on Balanced vs LongTail", "fig_E3_p99.png", e3)

    if {"variant","makespan"}.issubset(e4.columns):
        plot_e4("makespan", "makespan (ablation)", "fig_E4_makespan.png", e4)
    # if {"variant","p99"}.issubset(e4.columns):
    #     plot_e4("p99", "P99 (ablation)", "fig_E4_p99.png", e4)

if __name__ == "__main__":
    main()
    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")
