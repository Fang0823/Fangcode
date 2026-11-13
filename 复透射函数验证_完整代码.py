# -*- coding: utf-8 -*-
"""
复透射函数验证：实验 H_exp 与模型 H_model 对比

用途：对应论文新增小节“3.3.2 复透射函数验证”。

处理链：
NR.zip 原始 THz-TDS 频域复电场
    -> 每次扫描 100 个空间点复频谱平均
    -> 样品 / 空气参考，得到 H_exp(ω)
    -> 同一老化状态两次独立扫描做复数平均
    -> 温度-时间老化状态 S(T,t)
    -> Cole-Cole 复介电常数 ε*(ω,T,t)
    -> 理论复透射函数 H_model(ω)
    -> 幅值 rRMSE、相位 RMSE
    -> 12 个状态误差表
    -> 4 个代表状态幅值图、相位图
    -> 两张图横向拼接为论文图

注意：
1. 样品真实厚度 d = 2 mm。
2. 验证频段为 0.70–1.40 THz。
3. 当前理论 H_model 为不显式包含 Fabry–Pérot 多重反射项的单程传播表达式。
4. 相位只允许按整数倍 2π 调整分支，不加入任意常数相位修正。
5. 模型参数采用论文表2当前识别值，不在本脚本中重新拟合。

依赖：
    pip install numpy pandas matplotlib pillow

使用方法：
1. 将本脚本与 NR.zip 放在同一目录；
2. 运行：python 复透射函数验证_完整代码.py
3. 输出文件均保存在脚本同目录的 H_validation_output 文件夹。
"""

import io
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


# ============================================================
# 1. 文件与常数
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "NR.zip")
OUT_DIR = os.path.join(BASE_DIR, "H_validation_output")
os.makedirs(OUT_DIR, exist_ok=True)

C0 = 299_792_458.0       # 真空光速，m/s
D = 2.0e-3               # 样品真实厚度，m
R = 8.314462618           # 气体常数，J/(mol·K)

FREQ_MIN_THz = 0.70
FREQ_MAX_THz = 1.40

# 论文当前共享模型参数
Es = 41.2286e3            # J/mol
K_EPS = 0.24136           # 无量纲
DELTA_EPS_0 = 0.76427     # 无量纲
EPS_INF = 2.26403         # 无量纲
TAU_C = 0.003406e-12      # 0.003406 ps -> s
ALPHA_C = 0.60040         # Cole-Cole 分布参数

# 参考状态：120 ℃–72 h 对应 S=1
T_REF_K = 120.0 + 273.15
T_REF_H = 72.0


# ============================================================
# 2. 数据路径配置
#    每个老化状态含 2 次独立扫描；每次扫描含 100 个空间点
# ============================================================

CONFIGS = []


def add_config(condition, temp_c, aging_h, scan_path, ref_path):
    CONFIGS.append({
        "condition": condition,
        "temp_c": temp_c,
        "aging_h": aging_h,
        "scan_path": scan_path,
        "ref_path": ref_path,
    })


# 0 h
add_config("0 h", 0, 0, "NR/NEW-0h/0h-1", "NR/REF")
add_config("0 h", 0, 0, "NR/NEW-0h/0h-2", "NR/REF")

# 100 ℃：24 h 使用 NR/air，其余使用 NR/REF
for h in [24, 48, 72, 96]:
    ref = "NR/air" if h == 24 else "NR/REF"
    for j in [1, 2]:
        add_config(
            f"100 ℃–{h} h",
            100,
            h,
            f"NR/NEW-100C/100C_{h}/{h}-{j}",
            ref,
        )

# 120 ℃：使用 NR/air
for h in [24, 48, 72, 96]:
    for j in [1, 2]:
        add_config(
            f"120 ℃–{h} h",
            120,
            h,
            f"NR/NEW-120C/120C/120-{h}/120-{h}-{j}",
            "NR/air",
        )

# 140 ℃：使用 NR/REF
for h in [24, 48, 72]:
    for j in [1, 2]:
        add_config(
            f"140 ℃–{h} h",
            140,
            h,
            f"NR/NEW-140C/140C_{h}/{h}-{j}",
            "NR/REF",
        )


# ============================================================
# 3. 原始数据读取与复频谱平均
# ============================================================


def list_sample_dirs(all_names, base):
    """列出某次扫描下 Sample_* 子目录。"""
    prefix = base.rstrip("/") + "/"
    dirs = set()
    for name in all_names:
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        first = rel.split("/", 1)[0]
        if first.startswith("Sample_"):
            dirs.add(first)
    return sorted(dirs)


def read_csv_from_zip(zf, path):
    """从 zip 中读取一个数值 CSV。"""
    return np.loadtxt(io.BytesIO(zf.read(path)), delimiter=",")


def average_complex_fd(zf, all_names, base):
    """
    读取一次扫描下 100 个空间点的 FD.csv，并直接对复频谱做平均。

    假定 FD.csv 三列：
        第1列：频率 / THz
        第2列：复电场实部
        第3列：复电场虚部
    """
    sample_dirs = list_sample_dirs(all_names, base)
    if len(sample_dirs) != 100:
        raise RuntimeError(
            f"{base} 检测到 {len(sample_dirs)} 个 Sample_* 目录，预期为 100。"
        )

    freq = None
    spectra = []

    for d in sample_dirs:
        arr = read_csv_from_zip(zf, f"{base.rstrip('/')}/{d}/FD.csv")
        if freq is None:
            freq = arr[:, 0]
        spectra.append(arr[:, 1] + 1j * arr[:, 2])

    spectra = np.vstack(spectra)
    e_mean = np.mean(spectra, axis=0)
    return freq, e_mean


# ============================================================
# 4. 理论模型
# ============================================================


def aging_state_S(temp_c, aging_h):
    """温度-时间老化状态变量 S(T,t)。"""
    if aging_h == 0:
        return 0.0

    T = temp_c + 273.15
    return (aging_h / T_REF_H) * np.exp(
        -Es / R * (1.0 / T - 1.0 / T_REF_K)
    )


def model_epsilon(freq_THz, temp_c, aging_h):
    """基于老化状态的 Cole-Cole 复介电常数。"""
    omega = 2.0 * np.pi * freq_THz * 1e12
    S = aging_state_S(temp_c, aging_h)

    delta_eps = DELTA_EPS_0 * (1.0 + K_EPS * S)

    eps_complex = EPS_INF + delta_eps / (
        1.0 + (1j * omega * TAU_C) ** (1.0 - ALPHA_C)
    )
    return eps_complex


def model_transfer_function(freq_THz, temp_c, aging_h):
    """
    理论复透射函数：空气-样品-空气，单程传播，不显式加入 FP 项。

    H_model = 4*n~/(1+n~)^2 * exp[-i*ω*d/c*(n~-1)]
    """
    omega = 2.0 * np.pi * freq_THz * 1e12
    eps_complex = model_epsilon(freq_THz, temp_c, aging_h)

    n_complex = np.sqrt(eps_complex)

    H = (
        4.0 * n_complex / (1.0 + n_complex) ** 2
        * np.exp(-1j * omega * D / C0 * (n_complex - 1.0))
    )
    return H


# ============================================================
# 5. 相位分支处理
# ============================================================


def align_phase_by_2pi(H_exp, H_model):
    """
    分别展开实验和模型相位，只允许模型相位整体加减整数倍 2π。
    不加入任意相位偏置，不改变相位斜率。
    """
    phase_exp = np.unwrap(np.angle(H_exp))
    phase_model = np.unwrap(np.angle(H_model))

    k = int(np.round(np.mean(phase_exp - phase_model) / (2.0 * np.pi)))
    phase_model_aligned = phase_model + 2.0 * np.pi * k

    return phase_exp, phase_model_aligned, k


# ============================================================
# 6. 误差定义
# ============================================================


def amplitude_rrmse(H_exp, H_model):
    """
    幅值相对均方根误差：单位 %

    rRMSE_|H| = RMSE(|H_model|-|H_exp|) / mean(|H_exp|) * 100%
    """
    amp_exp = np.abs(H_exp)
    amp_model = np.abs(H_model)

    rmse = np.sqrt(np.mean((amp_model - amp_exp) ** 2))
    return rmse / np.mean(amp_exp) * 100.0


def phase_rmse(phase_exp, phase_model):
    """相位均方根误差：单位 rad。"""
    return np.sqrt(np.mean((phase_model - phase_exp) ** 2))


# ============================================================
# 7. 主计算：H_exp 与 H_model
# ============================================================


def calculate_validation():
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"找不到 {ZIP_PATH}")

    rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_names = zf.namelist()
        reference_cache = {}

        for cfg in CONFIGS:
            ref_path = cfg["ref_path"]

            # 同一空气参考只读取一次
            if ref_path not in reference_cache:
                reference_cache[ref_path] = average_complex_fd(
                    zf, all_names, ref_path
                )

            f_ref, E_ref = reference_cache[ref_path]
            f_sam, E_sam = average_complex_fd(
                zf, all_names, cfg["scan_path"]
            )

            if not np.allclose(f_ref, f_sam):
                raise RuntimeError(
                    f"样品与参考频率轴不一致：{cfg['scan_path']}"
                )

            H_exp = E_sam / E_ref

            band = (
                (f_sam >= FREQ_MIN_THz)
                & (f_sam <= FREQ_MAX_THz)
            )

            for f, h in zip(f_sam[band], H_exp[band]):
                rows.append({
                    "condition": cfg["condition"],
                    "temp_c": cfg["temp_c"],
                    "aging_h": cfg["aging_h"],
                    "scan": cfg["scan_path"].split("/")[-1],
                    "freq_THz": f,
                    "H_exp_re": h.real,
                    "H_exp_im": h.imag,
                })

    raw = pd.DataFrame(rows)

    # 两次独立扫描在复数域平均
    agg = (
        raw.groupby(
            ["condition", "temp_c", "aging_h", "freq_THz"],
            sort=False,
        )
        .agg(
            H_exp_re=("H_exp_re", "mean"),
            H_exp_im=("H_exp_im", "mean"),
        )
        .reset_index()
    )

    agg["H_exp"] = (
        agg["H_exp_re"].to_numpy()
        + 1j * agg["H_exp_im"].to_numpy()
    )

    agg["S"] = [
        aging_state_S(T, h)
        for T, h in zip(agg["temp_c"], agg["aging_h"])
    ]

    H_models = []
    for _, r in agg.iterrows():
        H_models.append(
            model_transfer_function(
                np.array([r["freq_THz"]]),
                r["temp_c"],
                r["aging_h"],
            )[0]
        )

    agg["H_model"] = H_models
    agg["H_exp_abs"] = np.abs(agg["H_exp"])
    agg["H_model_abs"] = np.abs(agg["H_model"])

    summary_rows = []

    for condition, g in agg.groupby("condition", sort=False):
        H_exp = g["H_exp"].to_numpy(dtype=complex)
        H_model = g["H_model"].to_numpy(dtype=complex)

        phase_exp, phase_model_aligned, k = align_phase_by_2pi(
            H_exp, H_model
        )

        amp_err = amplitude_rrmse(H_exp, H_model)
        phase_err = phase_rmse(phase_exp, phase_model_aligned)

        summary_rows.append({
            "老化状态": condition,
            "温度/℃": int(g["temp_c"].iloc[0]),
            "老化时间/h": int(g["aging_h"].iloc[0]),
            "S": float(g["S"].iloc[0]),
            "幅值rRMSE/%": amp_err,
            "相位RMSE/rad": phase_err,
            "相位2π分支k": k,
        })

        idx = g.index
        agg.loc[idx, "phase_exp_unwrapped"] = phase_exp
        agg.loc[idx, "phase_model_aligned"] = phase_model_aligned

    summary = pd.DataFrame(summary_rows)

    # 全部 12 个状态、全部频率点的总体误差
    H_exp_all = agg["H_exp"].to_numpy(dtype=complex)
    H_model_all = agg["H_model"].to_numpy(dtype=complex)

    global_amp_rrmse = amplitude_rrmse(H_exp_all, H_model_all)
    global_phase_rmse = np.sqrt(
        np.mean(
            (
                agg["phase_model_aligned"].to_numpy()
                - agg["phase_exp_unwrapped"].to_numpy()
            ) ** 2
        )
    )

    return agg, summary, global_amp_rrmse, global_phase_rmse


# ============================================================
# 8. 作图
#    为保证两张图横向拼接且不使用 subplot：
#    先分别生成两个独立 Figure，再用 Pillow 横向拼接。
# ============================================================


def set_chinese_font():
    """尽量自动选择可显示中文的字体。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


REPRESENTATIVE = [
    "0 h",
    "100 ℃–96 h",
    "120 ℃–96 h",
    "140 ℃–72 h",
]


def plot_amplitude(curves):
    fig = plt.figure(figsize=(8.2, 5.5))

    for condition in REPRESENTATIVE:
        g = curves[curves["condition"] == condition]
        plt.plot(
            g["freq_THz"],
            g["H_exp_abs"],
            label=f"{condition} 实验",
        )
        plt.plot(
            g["freq_THz"],
            g["H_model_abs"],
            label=f"{condition} 模型",
        )

    plt.xlabel("频率 / THz")
    plt.ylabel(r"复透射函数幅值 $|H(\omega)|$")
    plt.title("代表性老化状态下复透射函数幅值：实验值与模型值")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "复透射函数幅值_实验与模型.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_phase(curves):
    fig = plt.figure(figsize=(8.2, 5.5))

    for condition in REPRESENTATIVE:
        g = curves[curves["condition"] == condition]
        plt.plot(
            g["freq_THz"],
            g["phase_exp_unwrapped"],
            label=f"{condition} 实验",
        )
        plt.plot(
            g["freq_THz"],
            g["phase_model_aligned"],
            label=f"{condition} 模型",
        )

    plt.xlabel("频率 / THz")
    plt.ylabel("展开相位 / rad")
    plt.title("代表性老化状态下复透射函数相位：实验值与模型值")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "复透射函数相位_实验与模型.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def stitch_horizontal(left_path, right_path):
    """将两张独立图横向拼接。"""
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # 统一高度
    target_h = max(left.height, right.height)

    if left.height != target_h:
        new_w = int(left.width * target_h / left.height)
        left = left.resize((new_w, target_h), Image.LANCZOS)

    if right.height != target_h:
        new_w = int(right.width * target_h / right.height)
        right = right.resize((new_w, target_h), Image.LANCZOS)

    gap = 30
    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap, target_h),
        "white",
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))

    out_path = os.path.join(
        OUT_DIR,
        "复透射函数幅值与相位_横向组合.png",
    )
    canvas.save(out_path, dpi=(300, 300))
    return out_path


# ============================================================
# 9. 运行
# ============================================================


if __name__ == "__main__":
    set_chinese_font()

    curves, summary, global_amp, global_phase = calculate_validation()

    # 保存逐频率曲线数据
    curves_out = curves.drop(columns=["H_exp", "H_model"])
    curves_csv = os.path.join(OUT_DIR, "复透射函数验证_逐频率数据.csv")
    curves_out.to_csv(curves_csv, index=False, encoding="utf-8-sig")

    # 保存 12 个状态误差
    summary_csv = os.path.join(OUT_DIR, "复透射函数验证_12状态误差.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 单独作两张图，再横向拼接
    amp_png = plot_amplitude(curves)
    phase_png = plot_phase(curves)
    combo_png = stitch_horizontal(amp_png, phase_png)

    print("=" * 70)
    print("复透射函数验证完成")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print(f"全部状态总体幅值 rRMSE = {global_amp:.4f} %")
    print(f"全部状态总体相位 RMSE = {global_phase:.4f} rad")
    print(f"全部状态总体相位 RMSE = {np.degrees(global_phase):.4f}°")
    print()
    print("输出文件：")
    print(curves_csv)
    print(summary_csv)
    print(amp_png)
    print(phase_png)
    print(combo_png)
