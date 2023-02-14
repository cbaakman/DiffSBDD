import argparse
from pathlib import Path

import torch

import utils
from lightning_modules import LigandPocketDDPM
from dataset_pmhc import write_updated_peptide_coords_pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pdbfile", type=str)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--atom_level", type=bool, default=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)

    # load encoder and decoder
    encoder = model.pocket_type_encoder
    decoder = model.pocket_type_decoder

    atom_level = args.atom_level
    xh_peptide = model.generate_peptides(
        args.pdbfile,
        args.n_samples,
        timesteps=args.timesteps,
        atom_level=atom_level,
    )
    x_peptide = xh_peptide[:, : model.x_dims]
    #    atom_level = False if model.pocket_representation == "CA" else True

    size = int(len(x_peptide) / args.n_samples)

    for i in range(args.n_samples):
        write_updated_peptide_coords_pdb(
            x_peptide[i * size : (i + 1) * size],
            decoder,
            args.pdbfile,
            args.outdir / f"sample_{i}.pdb",
            atom_level=atom_level,
        )
