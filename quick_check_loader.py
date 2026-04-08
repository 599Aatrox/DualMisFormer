from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from data_provider.data_factory import data_provider

# 统一阈值
THRESH = 0.5
args = SimpleNamespace(
    root_path="./dataset/ETT-small/",  # 改成你自己的
    data_path="ETTm2.csv",  # 改成你自己的
    seq_len=96,
    label_len=48,
    pred_len=96,
    features="M",  # M/S/MS
    target="OT",  # 如果 features=S/MS 才需要对齐
    freq="h",
    cycle=24,
    batch_size=4,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.1,
    scale=True,
    data='custom',
    embed='timeF',
    task_name='long_term_forecast',
    seasonal_patterns="month",
)


def _get_cmap_and_norm(threshold: float):
    # 创建纯黑色和纯浅黄色的分段颜色映射
    colors = ['black', 'black', '#fbebdc', '#fbebdc']  # 低于阈值->黑色，高于阈值->浅黄
    cmap = LinearSegmentedColormap.from_list(
        "black_to_custom_yellow",
        colors,
        N=256
    )
    cmap = cmap.copy()
    # 阈值以下显示为黑色（under）
    cmap.set_under("black")

    # 只对 [threshold, 1] 映射为单一颜色；低于 threshold 的会用 under 颜色
    norm = Normalize(vmin=threshold, vmax=1.0, clip=False)
    return cmap, norm


def plot_var_corr_from_dataset(ds, max_vars=321, save_path="corr.png", threshold=THRESH):
    data = ds.data_x  # [T, N]
    if data.shape[1] > max_vars:
        data = data[:, :max_vars]

    C = np.abs(np.corrcoef(data, rowvar=False))  # [N, N]

    n = C.shape[0]
    avg_off = (C.sum() - np.trace(C)) / (n * (n - 1))
    print("Same-time Avg |corr| (off-diagonal) =", float(avg_off))

    cmap, norm = _get_cmap_and_norm(threshold)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(C, aspect="auto", cmap=cmap, norm=norm)
    plt.title(f"Variable Correlation of ETTm2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return avg_off


def _lagged_corr_matrix(data: np.ndarray, h: int) -> np.ndarray:
    H = data[:-h]
    F = data[h:]

    H = (H - H.mean(axis=0, keepdims=True)) / (H.std(axis=0, keepdims=True) + 1e-6)
    F = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-6)

    L = H.shape[0]
    return (H.T @ F) / max(L - 1, 1)


def plot_hist_future_corr_from_dataset(
        ds,
        pred_len: int,
        horizons=None,
        max_vars=321,
        save_path="hist_future_corr.png",
        use_abs=True,
        threshold=THRESH,
):
    data = ds.data_x  # [T, N]
    if data.shape[1] > max_vars:
        data = data[:, :max_vars]

    if horizons is None:
        horizons = [pred_len]
    elif isinstance(horizons, range):
        horizons = list(horizons)

    mats = []
    for h in horizons:
        if 0 < h < data.shape[0]:
            mats.append(_lagged_corr_matrix(data, h))

    C = np.mean(np.stack(mats, axis=0), axis=0)  # [N, N]
    if use_abs:
        C = np.abs(C)

    n = C.shape[0]
    avg_off = (C.sum() - np.trace(C)) / (n * (n - 1))
    print(f"Hist→Future Avg |corr| (off-diagonal), H={horizons} =", float(avg_off))

    cmap, norm = _get_cmap_and_norm(threshold)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(C, aspect="auto", cmap=cmap, norm=norm)
    plt.title(f"Hist→Future Variable Correlation of ETTm2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return avg_off


# ===== 运行检查 =====
ds, loader = data_provider(args, flag="train")
bx, by, bxm, bym, cyc = next(iter(loader))
print("batch_x:", bx.shape)  # [B, seq_len, N]
print("batch_y:", by.shape)  # [B, label_len+pred_len, N]
print("batch_x_mark:", bxm.shape)  # [B, seq_len, F]
print("batch_y_mark:", bym.shape)  # [B, label_len+pred_len, F]
print("cycle:", cyc.shape, cyc[:5])

# 1) 同时刻变量相关性（你的原图）
plot_var_corr_from_dataset(ds, save_path="corr.png", threshold=THRESH)

# 2) 历史→未来变量相关性（新增）
#   - 最标准：只看 h = pred_len
plot_hist_future_corr_from_dataset(
    ds,
    pred_len=args.pred_len,
    horizons=None,
    save_path=f"hist_future_h{args.pred_len}.png",
    threshold=THRESH,  # 这个一般要比0.8低，不然可能全黑；你也可以先设 None 看整体
)

#   - 可选：平均多个 horizon（更像论文）
# plot_hist_future_corr_from_dataset(
#     ds, pred_len=args.pred_len,
#     horizons=[24, 48, 96],
#     save_path="hist_future_mean.png",
#     threshold=0.3
# )
