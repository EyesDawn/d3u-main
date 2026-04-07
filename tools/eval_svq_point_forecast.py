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


def evaluate_dataset(dataset_name: str, output_root: Path, gpu: int, residual_space: str) -> dict:
    args = build_runtime_args(dataset_name, gpu)
    device = get_device(args)
    checkpoint_path = ROOT / "pretrain_checkpoints" / DATASET_CONFIGS[dataset_name]["setting"] / "checkpoint.pth"
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, predictions_path, _ = get_artifact_paths(dataset_dir, residual_space)

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


def plot_dataset(dataset_name: str, output_root: Path, residual_space: str) -> Path:
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
    space_label = metrics.get("space_label", get_space_label(residual_space))
    bins = min(80, max(20, int(np.sqrt(max(residual_last.size, 1)))))

    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(["#4C78A8", "#E45756", "#54A24B"])
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.patch.set_facecolor("#F7F7F5")
    ax.set_facecolor("#FCFCFB")

    kde_skipped = False
    residual_std = float(np.std(residual_last))
    kde_enabled = residual_last.size > 1 and residual_std > 1e-12
    sns.histplot(
        x=residual_last,
        bins=bins,
        stat="density",
        kde=False,
        color="#4C78A8",
        edgecolor="#FFFFFF",
        alpha=0.72,
        linewidth=0.7,
        ax=ax,
        label="Histogram",
    )
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="Zero Error")

    if kde_enabled:
        grid = np.linspace(residual_last.min(), residual_last.max(), 512)
        kde = gaussian_kde(residual_last)
        ax.plot(grid, kde(grid), color="#E45756", linewidth=2.4, label="KDE")
    else:
        kde_skipped = True

    residual_mean = float(np.mean(residual_last))
    residual_p05, residual_p95 = np.quantile(residual_last, [0.05, 0.95])

    ax.set_title(f"{dataset_name} Residual Distribution ({space_label}, Last Variable)", pad=14)
    ax.set_xlabel(f"Residual = y_true - y_hat ({space_label} space)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", color="#D9D9D4", alpha=0.55, linewidth=0.8)
    sns.despine(ax=ax, left=False, bottom=False)

    checkpoint_name = Path(metrics["checkpoint_path"]).parent.name
    meta = "\n".join(
        [
            f"space: {space_label}",
            "feature: last variable",
            f"mean: {residual_mean:.4f}",
            f"std: {residual_std:.4f}",
            f"p05/p95: {residual_p05:.4f} / {residual_p95:.4f}",
            "kde: skipped" if kde_skipped else "kde: enabled",
            f"ckpt: {checkpoint_name}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        meta,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {dataset_name}: saved {fig_path}")
    return fig_path


def main() -> None:
    cli_args = parse_args()
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
        plot_dataset(dataset_name, output_root, cli_args.residual_space)

    if summary:
        summary_path = output_root / f"summary_{cli_args.residual_space}.json"
        save_metrics(summary_path, summary)
        print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
