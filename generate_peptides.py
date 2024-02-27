import argparse
import io
from pathlib import Path
import random
import time

from Bio.PDB import PDBParser, PDBIO
import torch

import utils
from lightning_modules import LigandPocketDDPM
from dataset_pmhc import (
    write_updated_peptide_coords_pdb,
    process_pmhc_pdb_file,
    read_pdb_strings_hdf5_file,
    encode_types,
)


def combine_samples(peptide, mhc, n_samples):
    peptide = {
        "x": torch.cat([peptide["x"]] * n_samples, axis=0),
        "one_hot": torch.cat([peptide["one_hot"]] * n_samples, axis=0),
        "size": torch.tensor([peptide["size"]] * n_samples, device=peptide["x"].device),
        "mask": torch.repeat_interleave(
            torch.arange(n_samples).to(peptide["x"].device),
            peptide["size"],
        ),
    }
    mhc = {
        "x": torch.cat([mhc["x"]] * n_samples, axis=0),
        "one_hot": torch.cat([mhc["one_hot"]] * n_samples, axis=0),
        "size": torch.tensor([mhc["size"]] * n_samples, device=mhc["x"].device),
        "mask": torch.repeat_interleave(
            torch.arange(n_samples).to(mhc["x"].device),
            mhc["size"],
        ),
    }
    return peptide, mhc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pdbfile", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--atom_level", type=bool, default=False)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--return_frames", type=int, default=1)
    parser.add_argument("--select_interface", action="store_true", default=False)
    parser.add_argument("--structure_idx", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_level = args.atom_level

    # load model
    if args.data_dir is not None:
        model = LigandPocketDDPM.load_from_checkpoint(
            args.checkpoint, map_location=device, datadir=args.data_dir
        )
    else:
        model = LigandPocketDDPM.load_from_checkpoint(
            args.checkpoint, map_location=device
        )
    model = model.to(device)

    # load encoder and decoder
    encoder = model.pocket_type_encoder
    decoder = model.pocket_type_decoder

    if args.pdbfile.suffix == ".pdb":
        peptide, mhc = process_pmhc_pdb_file(
            args.pdbfile, atom_level=atom_level, device=device, select_interface=args.select_interface
        )
        name = args.pdbfile.stem
    else:
        pdb_string, names = read_pdb_strings_hdf5_file(args.pdbfile)
        pdb_stream = io.StringIO(pdb_string[0])
        peptide, mhc = process_pmhc_pdb_file(
            pdb_stream, atom_level=atom_level, device=device, select_interface=args.select_interface, pdb_string=pdb_string[args.structure_idx]
        )
        name = names[args.structure_idx]

    peptide["one_hot"] = encode_types(peptide["types"], encoder, device=device)
    mhc["one_hot"] = encode_types(mhc["types"], encoder, device=device)
    
    peptide, mhc = combine_samples(peptide, mhc, args.n_samples)
    print("average mhc size:", mhc["size"])

    start_time = time.time()
    xh_peptide = model.generate_peptides(
        peptide,
        mhc,
        timesteps=args.timesteps,
        return_frames=args.return_frames,
    )
    print("time taken:", time.time() - start_time)
    
    size = int(len(xh_peptide) / args.n_samples) if args.return_frames == 1 else int(len(xh_peptide[0]) / args.n_samples)

    samples_dir = args.outdir / name
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        sample_dir = samples_dir / f"{name}_sample_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        if args.return_frames == 1:
            x_peptide = xh_peptide[:, : model.x_dims]
            pdb_stream = io.StringIO(pdb_string[args.structure_idx]) if args.pdbfile.suffix != ".pdb" else args.pdbfile
            write_updated_peptide_coords_pdb(
                x_peptide[i * size : (i + 1) * size],
                decoder,
                pdb_stream,
                sample_dir / f"{name}_sample_{i}.pdb",
                atom_level=atom_level,
            )
        else:
            for j in range(args.return_frames):
                x_peptide = xh_peptide[j][:, : model.x_dims]
                pdb_stream = io.StringIO(pdb_string[args.structure_idx]) if args.pdbfile.suffix != ".pdb" else args.pdbfile
                write_updated_peptide_coords_pdb(
                    x_peptide[i * size : (i + 1) * size],
                    decoder,
                    pdb_stream,
                    sample_dir / f"{name}_sample_{i}_frame_{j}.pdb",
                    atom_level=atom_level,
                )

    # write original file to disk
    parser = PDBParser()
    if args.pdbfile.suffix == ".pdb":
        structure = parser.get_structure(name, args.pdbfile)
    else:
        structure = parser.get_structure(name, io.StringIO(pdb_string[args.structure_idx]))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(samples_dir / "original.pdb"))
