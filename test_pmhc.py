import argparse
import warnings
from pathlib import Path
import yaml
from time import time

import torch
import numpy as np
import pytorch_lightning as pl
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketDDPM
from dataset import ProcessedLigandPocketDataset
from equivariant_diffusion.en_diffusion import DistributionNodes

MAXITER = 10
MAXNTRIES = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--testfile", type=str, default="test.npz")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_time_batches", type=int, default=None)
    args = parser.parse_args()

    # load config of training
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.outdir.mkdir(exist_ok=True, parents=True)

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        datadir=args.datadir,
        outdir=args.outdir,
        device=device,

    )
    model = model.to(device)

    test_dataset = ProcessedLigandPocketDataset(Path(args.datadir, args.testfile))
    model.test_dataset = test_dataset

    histogram_file = Path(args.datadir, "size_distribution.npy")
    histogram = np.load(histogram_file).tolist()

    model.ddpm.size_distribution = DistributionNodes(histogram)

    trainer = pl.Trainer(
        devices=config["gpus"],
        accelerator="gpu",
        strategy=("ddp" if config["gpus"] > 1 else None),
    )
    model.test_timesteps = args.timesteps
    model.test_batch_size = args.batch_size
    model.test_n_samples = args.n_samples
    model.test_n_time_batches = args.n_time_batches
    trainer.test(model)
