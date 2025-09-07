import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from single_pixel import VFXNet, train_vfx_model
from image_utils import load_images
from experiments import EXPERIMENTS as DEFAULT_EXPERIMENTS
from piq import psnr
from job_runner import run_job_queue


STATIC_DIR = Path("/app/static")


@dataclass
class JobResult:
    name: str
    decoder_type: str
    model_type: str
    datasets: List[str]
    mean_bpp: float
    mean_psnr: float
    per_dataset: List[Dict[str, Any]]

    def to_row(self) -> List[str]:
        return [
            self.name,
            self.model_type,
            self.decoder_type,
            ";".join(self.datasets),
            f"{self.mean_bpp:.6f}",
            f"{self.mean_psnr:.6f}",
        ]


def _list_epoch_dirs(base_dir: Path) -> List[Tuple[int, Path]]:
    if not base_dir.exists():
        return []
    epochs = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                idx = int(p.name.split("_")[1])
                epochs.append((idx, p))
            except Exception:
                continue
    epochs.sort(key=lambda x: x[0])
    return epochs


def _evaluate_vfx_psnr(model: VFXNet, image_tensor: torch.Tensor, n_frames: Optional[int] = None) -> float:
    """
    Compute PSNR across frames.
    - Default: use all frames (no sampling).
    - If n_frames is provided and smaller than T, sample n_frames (same behavior as before).
    """
    T, H, W, C = image_tensor.shape
    device = image_tensor.device
    with torch.inference_mode():
        if n_frames is None or n_frames >= T:
            idx = torch.arange(T, device=device)
        else:
            # Keep prior behavior: random sample when enough frames, else linspace
            if T >= n_frames:
                idx = torch.randint(0, T, (n_frames,), device=device)
            else:
                idx = torch.linspace(0, max(T - 1, 0), steps=min(T, n_frames), device=device).round().to(torch.long)

        psnr_sum = 0.0
        count = 0
        for t_i in idx:
            # Use tensor time to satisfy full_image expectations
            t_tensor = t_i if isinstance(t_i, torch.Tensor) else torch.as_tensor(t_i, device=device)
            time_val = (t_tensor.float() / float(max(T, 1)))
            base = image_tensor[int(t_tensor.item())][..., :3].permute(2, 0, 1).unsqueeze(0)
            pred = model.full_image(time_val, H, W)[..., :3].permute(2, 0, 1).unsqueeze(0)
            pred = pred.to(device=base.device, dtype=base.dtype)
            psnr_val = psnr(base, pred).item()
            psnr_sum += float(psnr_val)
            count += 1
        return psnr_sum / max(count, 1)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sanitize_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "-" for c in name)


def run_single_job(
    job: Dict[str, Any],
    static_dir: Path,
    datasets_override: Optional[List[str]],
    epochs: int,
    bits_per_param: int,
    result_queue: Queue,
) -> None:
    try:
        name = job["name"]
        model_type = job.get("model_type", "vfx")
        if model_type != "vfx":
            raise ValueError(f"plot_scaling currently supports model_type='vfx' only (got {model_type})")

        cfg = dict(job.get("config", {}))  # shallow copy; we'll pop runtime keys
        decoder_type = cfg.get("decoder_type", "SpiralNet")
        device_str = cfg.get("device", "cuda")
        device = torch.device(device_str)

        # Resolve dataset list
        if datasets_override and len(datasets_override) > 0:
            dataset_list = datasets_override
        else:
            if "datasets" in job:
                if not isinstance(job["datasets"], list) or not job["datasets"]:
                    raise ValueError("'datasets' must be a non-empty list of dataset paths")
                dataset_list = job["datasets"]
            else:
                dataset_list = [job["dataset"]]

        per_dataset_metrics: List[Dict[str, Any]] = []

        for ds_rel in dataset_list:
            ds_path = static_dir / ds_rel
            ds_tag = Path(ds_rel).name
            exp_name = _sanitize_name(f"{name}-{ds_tag}")

            # Strip runtime-only keys for constructor
            ctor_cfg = {k: v for k, v in cfg.items() if k not in ("decoder_type", "device")}

            # 1) Train
            train_vfx_model(
                image_dir=ds_path,
                device=device,
                epochs=epochs,
                experiment_name=exp_name,
                decoder_type=decoder_type,
                decoder_config=dict(ctor_cfg),  # pass a copy
            )

            # 2) Load best/latest epoch weights
            anim_root = Path(f"anim_tests/{exp_name}")
            epoch_dirs = _list_epoch_dirs(anim_root)
            if not epoch_dirs:
                raise FileNotFoundError(f"No epoch folders found for experiment '{exp_name}' under {anim_root}")
            last_epoch, last_dir = epoch_dirs[-1]
            ckpt_path = last_dir / "model_weights.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            # 3) Load data + model
            frames = load_images(ds_path)
            T, H, W, _ = frames.shape
            model = VFXNet(H, W, device, decoder_type=decoder_type, decoder_config=dict(ctor_cfg))
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            # 4) Evaluate PSNR
            psnr_val = _evaluate_vfx_psnr(model, frames)

            # 5) Params → BPP
            num_params = _count_trainable_params(model)
            # Bits-per-pixel over the whole sequence (amortized across frames)
            bpp = (num_params * bits_per_param) / max(1, (H * W * T))

            per_dataset_metrics.append(
                {
                    "dataset": ds_rel,
                    "epoch": int(last_epoch),
                    "psnr": float(psnr_val),
                    "params": int(num_params),
                    "H": int(H),
                    "W": int(W),
                    "T": int(T),
                    "bpp": float(bpp),
                }
            )

            # Free memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Aggregate
        mean_psnr = sum(d["psnr"] for d in per_dataset_metrics) / len(per_dataset_metrics)
        mean_bpp = sum(d["bpp"] for d in per_dataset_metrics) / len(per_dataset_metrics)

        result = JobResult(
            name=name,
            decoder_type=decoder_type,
            model_type=model_type,
            datasets=dataset_list,
            mean_bpp=mean_bpp,
            mean_psnr=mean_psnr,
            per_dataset=per_dataset_metrics,
        )
        result_queue.put(result)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"error: {e}")
            print(f"OOM on {job.get('name','<unnamed>')}")
            sys.exit(42)
        raise


def _load_experiments(sweep_json: Optional[Path]) -> List[Dict[str, Any]]:
    if sweep_json is None:
        return list(DEFAULT_EXPERIMENTS)
    with open(sweep_json, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Sweep JSON must contain a list of experiment dicts")
        return data


def _write_csv(csv_path: Path, rows: List[JobResult]) -> None:
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "model_type", "decoder_type", "datasets", "mean_bpp", "mean_psnr"])
        for r in rows:
            writer.writerow(r.to_row())


def _make_plot(plot_path: Path, rows: List[JobResult]) -> None:
    import matplotlib.pyplot as plt
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    x = [r.mean_bpp for r in rows]
    y = [r.mean_psnr for r in rows]
    labels = [r.name for r in rows]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c="tab:blue")
    try:
        plt.xscale("log", base=2)
    except TypeError:
        # Fallback for older Matplotlib
        plt.xscale("log")
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Bits per pixel (from param count)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Bits per Pixel")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run a sweep and plot PSNR vs BPP (by param count)")
    parser.add_argument("--sweep-json", type=Path, default=None, help="Optional JSON file with experiments list")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/uvg/beauty",
        help="Single dataset subpath to use for all experiments (default for single-image scaling)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset subpaths to use for all experiments (overrides --dataset)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per experiment (choose a value that produces checkpoints)")
    parser.add_argument("--bits-per-param", type=int, default=32, help="Bits per trainable parameter (float32=32)")
    parser.add_argument("--csv-out", type=Path, default=Path("ai/results/scaling.csv"), help="Path to write results CSV")
    parser.add_argument("--plot-out", type=Path, default=Path("ai/results/scaling.png"), help="Path to save scatter plot")
    args = parser.parse_args()

    # Load sweep config
    experiments = _load_experiments(args.sweep_json)
    result_queue: Queue = Queue()
    results: List[JobResult] = []

    dataset_override = args.datasets if args.datasets else ([args.dataset] if args.dataset else None)

    def spawn(job: Dict[str, Any]) -> Process:
        p = Process(target=run_single_job, args=(job, STATIC_DIR, dataset_override, args.epochs, args.bits_per_param, result_queue))
        p.start()
        return p

    def poll_results() -> None:
        while not result_queue.empty():
            res = result_queue.get()
            if isinstance(res, JobResult):
                results.append(res)

    run_job_queue(experiments, spawn, poll_results=poll_results)

    # collect any straggler results
    while not result_queue.empty():
        res = result_queue.get()
        if isinstance(res, JobResult):
            results.append(res)

    # Sort by BPP for nicer plotting
    results.sort(key=lambda r: r.mean_bpp)

    # Write CSV + plot
    csv_out = Path("/app") / args.csv_out if not args.csv_out.is_absolute() else args.csv_out
    plot_out = Path("/app") / args.plot_out if not args.plot_out.is_absolute() else args.plot_out

    _write_csv(csv_out, results)
    _make_plot(plot_out, results)

    print(f"Wrote CSV to {csv_out}")
    print(f"Saved plot to {plot_out}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
