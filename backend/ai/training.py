import sys
from pathlib import Path
from multiprocessing import Process

import multiprocessing as mp

import torch

from single_pixel import train_vfx_model
from gauges import train_drill_model
from experiments import EXPERIMENTS
from job_runner import run_job_queue


def run_single_job(job, static_dir):
    try:
        name = job["name"]
        dataset_path = static_dir / job["dataset"]
        model_type = job.get("model_type", "vfx")
        if model_type == "vfx":
            train_job = train_vfx_model
        elif model_type == "drill":
            train_job = train_drill_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        decoder_config = job["config"]
        print(f"Starting VFX Model: {name}")
        train_job(
            image_dir=dataset_path,
            device=torch.device(decoder_config.pop("device", "cuda")),
            experiment_name=name,
            decoder_type=decoder_config.pop("decoder_type"),
            decoder_config=decoder_config,
        )
        print(f"Finished: {name}")
        return "done"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"error: {e}")
            print(f"OOM on {name}")
            sys.exit(42)
        raise


def main():
    STATIC_DIR = Path("/app/static")

    def spawn(job):
        p = Process(target=run_single_job, args=(job, STATIC_DIR))
        p.start()
        return p

    run_job_queue(EXPERIMENTS, spawn)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
