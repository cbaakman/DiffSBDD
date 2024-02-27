import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import warnings
from Bio import BiopythonDeprecationWarning

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from dataset_pmhc import (
    get_encoder_decoder,
    process_pmhc_pdb_file,
    process_pmhc_hdf5_file,
    encode_types,
)
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

def process_pmhc_directory(dir_path: Path, atom_level=True, encoder=None, decoder=None, ids_to_keep=None, select_interface=False):
    """
    Process the directory to get the peptide and pocket representations
    :param dir_path: the path of the directory
    :param kwargs: the arguments to pass to process_pmhc_pdb_file
    :return: peptide, mhc
    """
    # encoder, decoder = get_encoder_decoder(dir_path, atom_level=atom_level)
    peptide_list = []
    mhc_list = []
    types = set()
    complex_names = []
    for pdb_file in os.listdir(dir_path):
        pdb_path = dir_path / pdb_file
        if pdb_path.suffix == ".pdb":
            if ids_to_keep is not None:
                if pdb_path.stem not in ids_to_keep:
                    continue
            peptide, mhc = process_pmhc_pdb_file(pdb_path, atom_level=atom_level, select_interface=select_interface)
            peptides = [peptide]
            mhcs = [mhc]
            names = [pdb_path.stem]
            # peptide.append(peptide_)
            # mhc.append(mhc_)
            # types.update(peptide_["type"])
            # types.update(mhc_["type"])
            # complex_names.append(pdb_path.stem)

        elif pdb_path.suffix == ".hdf5":
            peptides, mhcs, names = process_pmhc_hdf5_file(
                pdb_path, atom_level=atom_level, select_interface=select_interface
            )
        else:
            continue

        for peptide, mhc, name in zip(peptides, mhcs, names):
            complex_names.append(name)
            peptide_list.append(peptide)
            mhc_list.append(mhc)
            types.update(peptide["types"])
            types.update(mhc["types"])

    if encoder is None or decoder is None:
        encoder, decoder = get_encoder_decoder(types)
    # peptide["one_hot"] = encode_types(pep_types, encoder)
    # mhc["one_hot"] = encode_types(mhc_types, encoder)

    # peptide["mask"] = torch.repeat_interleave(
    #     torch.arange(peptide["x"].shape[0]),
    #     peptide["size"],
    #  )
    # mhc["mask"] = torch.repeat_interleave(
    #     torch.arange(mhc["x"].shape[0]),
    #     mhc["size"],
    # )

    return peptide_list, mhc_list, encoder, decoder, complex_names


def process_save_pdb_dir(
    pdb_dir,
    outdir,
    atom_level,
    encoder_decoder_dir=None,
    train_frac=0.6,
    val_frac=0.2,
    ids_to_keep=None,
    select_interface=False,
    group_sequences=False,
    seed=42,
    save_path=None,
    load_path=None,
):
    input_encoder = None
    input_decoder = None
    if encoder_decoder_dir is not None:
        with open(Path(encoder_decoder_dir) / "encoder.json", "r") as f:
            input_encoder = json.load(f)
        with open(Path(encoder_decoder_dir) / "decoder.json", "r") as f:
            input_decoder = json.load(f)

    if load_path is None:    
        peptides, mhcs, encoder, decoder, names = process_pmhc_directory(
            Path(pdb_dir),
            atom_level=atom_level,
            encoder=input_encoder,
            decoder=input_decoder,
            ids_to_keep=ids_to_keep,
            select_interface=select_interface,
        )
    else:
        with open(load_path, "rb") as f:
            datadict = torch.load(f)
        peptides = datadict["peptides"]
        mhcs = datadict["mhcs"]
        names = datadict["names"]
        encoder = input_encoder
        decoder = input_decoder

    with open(Path(outdir) / "encoder.json", "w") as f:
        json.dump(encoder, f)

    with open(Path(outdir) / "decoder.json", "w") as f:
        json.dump(decoder, f)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        savedict = {
            "peptides": peptides,
            "mhcs": mhcs,
            "names": names,
        }
        with open(save_path, "wb") as f:
            torch.save(savedict, f)

    n = len(peptides)
    if group_sequences:
        # obtain a sequence id per peptide
        peptide_sequence_idx = group_peptide_sequences(peptides)

        # split train, val and test integers in such a way that it is
        # random but the same sequence id is not in train and val or test
        # at the same time
        train_idx, val_idx, test_idx = attribute_data_to_splits_selective(
            peptide_sequence_idx, train_frac, val_frac, seed=seed
        )
    else:
        # randomly split the data into train, val, and test
        n = len(peptides)
        train_size = int(n * train_frac)
        val_size = int(n * val_frac)

        train_idx = np.random.choice(n, train_size, replace=False)
        val_idx = np.random.choice(
            np.setdiff1d(np.arange(n), train_idx), val_size, replace=False
        )
        test_idx = np.setdiff1d(np.arange(n), np.concatenate((train_idx, val_idx)))

    for idx, split in zip(
        [train_idx, val_idx, test_idx, np.arange(n)], ["train", "val", "test", "all"]
    ):
        path = Path(outdir) / f"{split}.npz"
        
        peptide, mhc = combine_samples(peptides, mhcs, idx)

        pep_one_hot = encode_types(peptide["types"], encoder)
        mhc_one_hot = encode_types(mhc["types"], encoder)

        mhc_mask = np.repeat(np.arange(len(mhc["size"]), dtype=np.int32), mhc["size"])
        pep_mask = np.repeat(
            np.arange(len(peptide["size"]), dtype=np.int32), peptide["size"]
        )
        split_names = [names[i] for i in idx]

        np.savez(
            path,
            names=split_names,
            lig_coords=peptide["x"],
            lig_one_hot=pep_one_hot.numpy(),
            lig_mask=pep_mask,
            pocket_c_alpha=mhc["x"],
            pocket_one_hot=mhc_one_hot.numpy(),
            pocket_mask=mhc_mask,
        )
        if split == "all":
            n_nodes = get_n_nodes(pep_mask, mhc_mask, smooth_sigma=1.0)
            np.save(Path(outdir, "size_distribution.npy"), n_nodes)

        # print stats for split
        print(f"{split} set:")
        print(f"  {len(idx)} complexes")

        print(
            f"  {len(peptide['size'])} peptides of average length {np.mean(peptide['size'])}"
        )
        print(f"  {peptide['x'].shape} peptide coordinates shape")
        print(f"  {pep_one_hot.shape} peptide one hot shape")
        print(f"  {pep_mask} peptide mask")

        print(f"  {len(mhc['size'])} mhc of average length {np.mean(mhc['size'])}")
        print(f"  {mhc['x'].shape} mhc coordinates shape")
        print(f"  {mhc_one_hot.shape} mhc one hot shape")
        print(f"  {mhc_mask} mhc mask shape")


def combine_samples(peptides, mhcs, idx):
    peptide = {
        "x": np.concatenate([peptides[i]["x"].numpy() for i in idx], axis=0),
        "types": [node for i in idx for node in peptides[i]["types"]],
        "size": [peptides[i]["size"] for i in idx],
    }
    mhc = {
        "x": np.concatenate([mhcs[i]["x"].numpy() for i in idx], axis=0),
        "types": [node for i in idx for node in mhcs[i]["types"]],
        "size": [mhcs[i]["size"] for i in idx],
    }
    return peptide, mhc


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

def group_peptide_sequences(peptides):
    """
    Groups the peptides with the exact same sequence together
    so that they can be added to the same split to prevent
    data leakage.
    """
    peptide_sequences = ["".join(pep["types"]) for pep in peptides]
    unique_sequences = np.unique(peptide_sequences)
    sequence_to_idx = {seq: i for i, seq in enumerate(unique_sequences)}
    peptide_idx = [sequence_to_idx[seq] for seq in peptide_sequences]
    return peptide_idx

def attribute_data_to_splits(group_indices, train_fraction, val_fraction, seed=42):
    total_datapoints = len(group_indices)
    train_count = int(train_fraction * total_datapoints)
    val_count = int(val_fraction * total_datapoints)

    group_data = defaultdict(list)
    for idx, group_id in enumerate(group_indices):
        group_data[group_id].append(idx)

    random.seed(seed)
    unique_group_ids = list(group_data.keys())
    random.shuffle(unique_group_ids)

    train_indices = []
    val_indices = []
    test_indices = []

    for group_id in unique_group_ids:
        if len(train_indices) < train_count:
            train_indices += group_data[group_id]
        elif len(val_indices) < val_count:
            val_indices += group_data[group_id]
        else:
            test_indices += group_data[group_id]

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def attribute_data_to_splits_selective(group_indices, train_fraction, val_fraction, allowed_spillover=100, seed=42):
    """
    Version of the `attribute_data_to_splits` function that ensures that
    splits do not get too large by not greadily adding groups to the splits
    but instead keeping the remaining space in each split into account.

    The `allowed_spillover` parameter determines how many datapoints can be added
    over the limit before the group is skipped.
    """
    total_datapoints = len(group_indices)
    train_count = int(train_fraction * total_datapoints)
    val_count = int(val_fraction * total_datapoints)

    group_data = defaultdict(list)
    for idx, group_id in enumerate(group_indices):
        group_data[group_id].append(idx)

    random.seed(seed)
    unique_group_ids = list(group_data.keys())
    random.shuffle(unique_group_ids)

    train_indices = []
    val_indices = []
    test_indices = []

    for group_id in unique_group_ids:
        if len(train_indices) < train_count and len(train_indices) + len(group_data[group_id]) <= train_count + allowed_spillover:
            train_indices += group_data[group_id]
        elif len(val_indices) < val_count and len(val_indices) + len(group_data[group_id]) <= val_count + allowed_spillover:
            val_indices += group_data[group_id]
        elif len(test_indices) + len(group_data[group_id]) <= allowed_spillover:
            test_indices += group_data[group_id]
        else:
            # if we are left with a group that is too large, add it to the group
            # with the largest difference to the target size
            train_diff = train_count - len(train_indices)
            val_diff = val_count - len(val_indices)
            test_diff = total_datapoints - train_count - val_count - len(test_indices)

            diffs = [train_diff, val_diff, test_diff]
            max_diff = max(diffs)
            max_diff_idx = diffs.index(max_diff)

            if max_diff_idx == 0:
                train_indices += group_data[group_id]
            elif max_diff_idx == 1:
                val_indices += group_data[group_id]
            else:
                test_indices += group_data[group_id]

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_dir", type=str, help="path to pdb files")
    parser.add_argument("save_dir", type=Path, help="path to save processed files")
    parser.add_argument(
        "--atom_level", type=bool, default=False, help="whether to use atom level"
    )
    parser.add_argument("--encoder_decoder_dir", type=Path, required=True, default=None)
    parser.add_argument("--id_json_path", type=Path, required=False, default=None)
    parser.add_argument("--select_interface", action="store_true", default=False)
    parser.add_argument("--group_sequences", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=0.6)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--load_path", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    ids_to_process = None
    if args.id_json_path is not None:
        with open(args.id_json_path, "r") as f:
            ids_to_process = json.load(f)

    process_save_pdb_dir(
        args.pdb_dir,
        args.save_dir,
        args.atom_level,
        encoder_decoder_dir=args.encoder_decoder_dir,
        ids_to_keep=ids_to_process,
        select_interface=args.select_interface,
        group_sequences=args.group_sequences,
        train_frac=args.train_fraction,
        val_frac=args.val_fraction,
        save_path=args.save_path,
        load_path=args.load_path,
    )