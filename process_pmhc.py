import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from dataset_pmhc import get_encoder_decoder, process_pmhc_pdb_file


def process_pmhc_directory(dir_path, atom_level=True):
    """
    Process the directory to get the peptide and pocket representations
    :param dir_path: the path of the directory
    :param kwargs: the arguments to pass to process_pmhc_pdb_file
    :return: peptide, mhc
    """
    encoder, decoder = get_encoder_decoder(dir_path, atom_level=atom_level)
    peptide = {
        "x": torch.Tensor(),
        "one_hot": torch.Tensor(),
        "size": torch.Tensor(),
        "mask": torch.Tensor(),
    }
    mhc = {
        "x": torch.Tensor(),
        "one_hot": torch.Tensor(),
        "size": torch.Tensor(),
        "mask": torch.Tensor(),
    }
    names = []
    for pdb_file in os.listdir(dir_path):
        pdb_path = dir_path / pdb_file
        if pdb_path.suffix == ".pdb":
            names.append(pdb_path.stem)
            peptide_, mhc_ = process_pmhc_pdb_file(
                Path(dir_path) / pdb_file, encoder, atom_level=atom_level
            )
            peptide["x"] = torch.cat([peptide["x"], peptide_["x"]], dim=0)
            peptide["one_hot"] = torch.cat(
                [peptide["one_hot"], peptide_["one_hot"]], dim=0
            )
            peptide["size"] = torch.cat((peptide["size"], peptide_["size"]))
            peptide["mask"] = torch.cat((peptide["mask"], peptide_["mask"]))

            mhc["x"] = torch.cat([mhc["x"], mhc_["x"]], dim=0)
            mhc["one_hot"] = torch.cat([mhc["one_hot"], mhc_["one_hot"]], dim=0)
            mhc["size"] = torch.cat((mhc["size"], mhc_["size"]))
            mhc["mask"] = torch.cat((mhc["mask"], mhc_["mask"]))

    return peptide, mhc, encoder, decoder, names


def process_save_pdb_dir(
    pdb_dir,
    outdir,
    atom_level,
):
    peptide, mhc, encoder, decoder, names = process_pmhc_directory(
        Path(pdb_dir),
        atom_level=atom_level,
    )
    with open(Path(outdir) / "encoder.json", "w") as f:
        json.dump(encoder, f)

    with open(Path(outdir) / "decoder.json", "w") as f:
        json.dump(decoder, f)

    for split in ["train", "val", "test"]:
        path = Path(outdir) / f"{split}.npz"

        np.savez(
            path,
            names=names,
            lig_coords=peptide["x"].cpu().numpy(),
            lig_one_hot=peptide["one_hot"].cpu().numpy(),
            lig_mask=peptide["mask"].cpu().numpy(),
            pocket_c_alpha=mhc["x"].cpu().numpy(),
            pocket_one_hot=mhc["one_hot"].cpu().numpy(),
            pocket_mask=mhc["mask"].cpu().numpy(),
        )

    n_nodes = get_n_nodes(peptide["mask"], mhc["mask"], smooth_sigma=1.0)
    np.save(Path(outdir, "size_distribution.npy"), n_nodes)


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1, np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(
        f"Original histogram: {np.count_nonzero(joint_histogram)}/"
        f"{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled"
    )

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram,
            sigma=smooth_sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )

        print(
            f"Smoothed histogram: {np.count_nonzero(filtered_histogram)}/"
            f"{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled"
        )

        joint_histogram = filtered_histogram

    return joint_histogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_dir", type=str, help="path to pdb files")
    parser.add_argument("save_dir", type=str, help="path to save processed files")
    parser.add_argument(
        "--atom_level", type=bool, default=False, help="whether to use atom level"
    )
    args = parser.parse_args()
    process_save_pdb_dir(
        args.pdb_dir,
        args.save_dir,
        args.atom_level,
    )
