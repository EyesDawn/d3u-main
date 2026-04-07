from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from model9_NS_transformer.condition_models import SVQ
from utils.metrics import metric


COMMON_CONFIG = {
    "model": "SVQ",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 192,
    "features": "M",
    "target": "OT",
    "embed": "timeF",
    "freq": "h",
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_size": 16,
    "stride": 8,
    "padding_patch": "end",
    "individual": 0,
    "sout": 0,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "kernel_size": 15,
    "fourier_factor": 1.0,
    "svq": 1,
    "wFFN": 0,
    "codebook_size": 256,
    "length": 96,
    "decomposition": False,
    "use_uncertainty": True,
    "use_gpu": True,
    "use_multi_gpu": False,
    "devices": "0",
    "num_workers": 0,
    "batch_size": 128,
    "test_batch_size": 32,
    "seed": 2021,
}


DATASET_CONFIGS = {
    "ETTh1": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_ETTh1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "ETTh1",
        "data_name": "ETTh1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model_c": 512,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 1,
        "d_model_d": 128,
        "num_codebook": 1,
        "test_batch_size": 64,
    },
    "ETTh2": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_ETTh2_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "ETTh2",
        "data_name": "ETTh2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model_c": 512,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 1,
        "d_model_d": 128,
        "num_codebook": 1,
        "test_batch_size": 64,
    },
    "ETTm1": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_ETTm1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "ETTm1",
        "data_name": "ETTm1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model_c": 512,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 1,
        "d_model_d": 128,
        "num_codebook": 1,
        "test_batch_size": 64,
    },
    "ETTm2": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_ETTm2_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "ETTm2",
        "data_name": "ETTm2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model_c": 512,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 1,
        "d_model_d": 128,
        "num_codebook": 1,
        "test_batch_size": 64,
    },
    "weather": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_weather_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "custom",
        "data_name": "weather",
        "root_path": "./dataset/",
        "data_path": "weather.csv",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "d_model_c": 256,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 1,
        "d_model_d": 256,
        "num_codebook": 1,
        "test_batch_size": 4,
    },
    "ECL": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_ECL_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "custom",
        "data_name": "ECL",
        "root_path": "./dataset/",
        "data_path": "electricity.csv",
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        "d_model_c": 512,
        "n_heads_c": 8,
        "e_layers_c": 2,
        "d_layers_c": 1,
        "d_ff": 512,
        "depth": 2,
        "d_model_d": 128,
        "num_codebook": 2,
        "test_batch_size": 1,
    },
    "traffic": {
        "setting": "False_ts100_PatchDN_96_192_SVQ_traffic_ftM_sl96_ll48_pl192_dm256_nh16_el3_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0",
        "data": "custom",
        "data_name": "traffic",
        "root_path": "./dataset/",
        "data_path": "traffic.csv",
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
        "d_model_c": 256,
        "n_heads_c": 16,
        "e_layers_c": 3,
        "d_layers_c": 1,
        "d_ff": 256,
        "depth": 1,
        "d_model_d": 128,
        "num_codebook": 2,
        "codebook_size": 512,
        "batch_size": 16,
        "test_batch_size": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SVQ point forecasts from pretrain_checkpoints and plot residual distributions."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS.keys()),
        default=["ETTm1", "weather", "ECL"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "svq_eval"),
        help="Directory used to store metrics, predictions, and figures.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--residual-space",
        choices=["zscore", "original"],
        default="zscore",
        help="Export and plot residuals in StandardScaler space or inverse-transformed original space.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model evaluation and regenerate plots from saved predictions.",
    )
    parser.add_argument(
        "--clip_min",
        "--clip-min",
        dest="clip_min",
        type=float,
        default=None,
        help="Lower clipping bound applied to residuals before plotting only.",
    )
    parser.add_argument(
        "--clip_max",
        "--clip-max",
        dest="clip_max",
        type=float,
        default=None,
        help="Upper clipping bound applied to residuals before plotting only.",
    )
    parser.add_argument(
        "--no_clip",
        "--no-clip",
        dest="no_clip",
        action="store_true",
        help="Disable clipping even if clip_min or clip_max are provided.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime_args(dataset_name: str, gpu: int) -> SimpleNamespace:
    config = dict(COMMON_CONFIG)
    config.update(DATASET_CONFIGS[dataset_name])
    config["gpu"] = gpu
    config["use_gpu"] = bool(torch.cuda.is_available())
    config["root_path"] = str((ROOT / config["root_path"]).resolve()) + "/"
    return SimpleNamespace(**config)


def get_device(args: SimpleNamespace) -> torch.device:
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    return device


def ensure_serializable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [ensure_serializable(item) for item in value]
    if isinstance(value, list):
        return [ensure_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: ensure_serializable(item) for key, item in value.items()}
    return value


def save_metrics(metrics_path: Path, metrics: dict) -> None:
    metrics_path.write_text(
        json.dumps(ensure_serializable(metrics), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[warn] checkpoint load mismatch for {checkpoint_path.name}: "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )


def get_artifact_paths(dataset_dir: Path, residual_space: str) -> tuple[Path, Path, Path]:
    suffix = residual_space.lower()
    return (
        dataset_dir / f"metrics_{suffix}.json",
        dataset_dir / f"predictions_{suffix}.npz",
        dataset_dir / f"residual_{dataset_dir.name}_{suffix}_hist_kde.png",
    )


def inverse_transform_forecast(data_set, values: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    flat_values = values.reshape(-1, original_shape[-1])
    restored = data_set.inverse_transform(flat_values)
    return restored.reshape(original_shape)


def get_space_label(residual_space: str) -> str:
    return "z-score" if residual_space == "zscore" else "original"


def format_clip_bound(bound: float) -> str:
    if np.isneginf(bound):
        return "-inf"
    if np.isposinf(bound):
        return "+inf"
    return f"{bound:g}"


def clip_residuals_for_plot(
    residuals: np.ndarray,
    clip_min: float | None,
    clip_max: float | None,
    no_clip: bool,
) -> tuple[np.ndarray, bool, str]:
    if no_clip or (clip_min is None and clip_max is None):
        return residuals, False, "no clipping"

    lower = float(clip_min) if clip_min is not None else float("-inf")
    upper = float(clip_max) if clip_max is not None else float("inf")
    if lower > upper:
        raise ValueError(f"clip_min must be <= clip_max, got {clip_min} > {clip_max}")

    clipped = np.clip(residuals, lower, upper)
    clip_note = f"clipped to [{format_clip_bound(lower)}, {format_clip_bound(upper)}]"
    return clipped, True, clip_note


def darken_color(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def evaluate_dataset(dataset_name: str, output_root: Path, gpu: int, residual_space: str) -> dict:
    args = build_runtime_args(dataset_name, gpu)
    device = get_device(args)
    checkpoint_path = ROOT / "pretrain_checkpoints" / DATASET_CONFIGS[dataset_name]["setting"] / "checkpoint.pth"
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, predictions_path, _ = get_artifact_paths(dataset_dir, residual_space)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {dataset_name}: {checkpoint_path}. "
            f"Expected setting={DATASET_CONFIGS[dataset_name]['setting']}"
        )

    print(f"[eval] {dataset_name}: loading checkpoint {checkpoint_path}")
    model = SVQ.Model(args).float().to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_data, test_loader = data_provider(args, flag="test")
    y_true_batches = []
    y_hat_batches = []

    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            y_hat, _, _, _ = model(batch_x, None, None, None)
            y_true = batch_y[:, -args.pred_len :, :]
            y_true_batches.append(y_true.detach().cpu().numpy())
            y_hat_batches.append(y_hat.detach().cpu().numpy())

    y_true_all = np.concatenate(y_true_batches, axis=0)
    y_hat_all = np.concatenate(y_hat_batches, axis=0)

    if residual_space == "original":
        y_true_export = inverse_transform_forecast(test_data, y_true_all)
        y_hat_export = inverse_transform_forecast(test_data, y_hat_all)
    else:
        y_true_export = y_true_all
        y_hat_export = y_hat_all

    residual_all = y_true_export - y_hat_export

    mae, mse, rmse, mape, mspe = metric(y_hat_export, y_true_export)
    np.savez_compressed(
        predictions_path,
        y_true=y_true_export,
        y_hat=y_hat_export,
        residual=residual_all,
        y_true_last=y_true_export[:, :, -1],
        y_hat_last=y_hat_export[:, :, -1],
        residual_last=residual_all[:, :, -1],
    )

    metrics = {
        "dataset": dataset_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "num_windows": int(y_true_export.shape[0]),
        "pred_len": int(y_true_export.shape[1]),
        "n_vars": int(y_true_export.shape[2]),
        "prediction_file": str(predictions_path.resolve()),
        "space": residual_space,
        "space_label": get_space_label(residual_space),
        "residual_definition": "y_true - y_hat",
        "feature_for_plot": "last variable",
    }
    save_metrics(metrics_path, metrics)
    print(
        f"[eval] {dataset_name}: space={residual_space}, mae={metrics['mae']:.6f}, "
        f"mse={metrics['mse']:.6f}, rmse={metrics['rmse']:.6f}, windows={metrics['num_windows']}"
    )
    return metrics


def plot_dataset(
    dataset_name: str,
    output_root: Path,
    residual_space: str,
    clip_min: float | None,
    clip_max: float | None,
    no_clip: bool,
) -> Path:
    dataset_dir = output_root / dataset_name
    metrics_path, predictions_path, fig_path = get_artifact_paths(dataset_dir, residual_space)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {predictions_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("space") != residual_space:
        raise ValueError(
            f"Residual space mismatch for {dataset_name}: expected {residual_space}, found {metrics.get('space')}"
        )
    residual_last = np.load(predictions_path)["residual_last"].reshape(-1)
    residual_plot, _, clip_note = clip_residuals_for_plot(
        residual_last,
        clip_min,
        clip_max,
        no_clip,
    )
    space_label = metrics.get("space_label", get_space_label(residual_space))
    bins = min(72, max(24, int(np.sqrt(max(residual_plot.size, 1)))))

    sns.set_theme(
        style="white",
        context="paper",
        rc={
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
        },
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    base_color = "tab:orange"
    kde_color = darken_color(base_color, factor=0.72)
    zero_line_color = "#4A4A4A"

    residual_std = float(np.std(residual_plot))
    kde_enabled = residual_plot.size > 1 and residual_std > 1e-12
    sns.histplot(
        x=residual_plot,
        bins=bins,
        stat="density",
        kde=False,
        color=base_color,
        alpha=0.40,
        edgecolor=None,
        ax=ax,
        label="Histogram",
    )
    ax.axvline(0.0, color=zero_line_color, linestyle="--", linewidth=1.1, label="Zero Residual")

    if kde_enabled:
        grid = np.linspace(residual_plot.min(), residual_plot.max(), 512)
        kde = gaussian_kde(residual_plot)
        ax.plot(grid, kde(grid), color=kde_color, linewidth=2.2, label="KDE")

    ax.set_title(dataset_name, loc="left", pad=8, fontweight="semibold")
    ax.text(
        0.0,
        1.01,
        f"Residual distribution ({space_label} space, last variable)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#5A5A5A",
    )
    ax.text(
        1.0,
        1.01,
        clip_note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#6A6A6A",
    )

    ax.set_xlabel("Residual = y_true - y_hat")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="#D8D8D8", alpha=0.8, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.02)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="#3A3A3A")
    ax.legend(loc="upper right", frameon=False, handlelength=2.4)
    sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] {dataset_name}: saved {fig_path}")
    return fig_path


def main() -> None:
    cli_args = parse_args()
    if (
        not cli_args.no_clip
        and cli_args.clip_min is not None
        and cli_args.clip_max is not None
        and cli_args.clip_min > cli_args.clip_max
    ):
        raise ValueError(f"clip_min must be <= clip_max, got {cli_args.clip_min} > {cli_args.clip_max}")

    output_root = Path(cli_args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(COMMON_CONFIG["seed"])

    summary = {}
    for dataset_name in cli_args.datasets:
        if not cli_args.plot_only:
            summary[dataset_name] = evaluate_dataset(
                dataset_name,
                output_root,
                cli_args.gpu,
                cli_args.residual_space,
            )
        else:
            metrics_path, _, _ = get_artifact_paths(output_root / dataset_name, cli_args.residual_space)
            if metrics_path.exists():
                summary[dataset_name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_dataset(
            dataset_name,
            output_root,
            cli_args.residual_space,
            cli_args.clip_min,
            cli_args.clip_max,
            cli_args.no_clip,
        )

    if summary:
        summary_path = output_root / f"summary_{cli_args.residual_space}.json"
        save_metrics(summary_path, summary)
        print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
