import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from Bio.PDB import PDBParser, PDBIO

# from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
from Bio.PDB.Polypeptide import three_to_one

from pathlib import Path

INT_TYPE = torch.int64
FLOAT_TYPE = torch.float32


def get_encoder_decoder(pdb_directory, atom_level=True):
    """
    Get the encoder and decoder for all chains in the directory
    :param pdb_directory: the directory containing the pdb files
    :param atom_level: whether to use atoms or residues

    :return: encoder, decoder
    """
    chains = []
    for pdb_file in os.listdir(pdb_directory):
        pdb_path = pdb_directory / pdb_file
        parser = PDBParser(QUIET=True)
        pdb_models = parser.get_structure("", pdb_path)
        chains += [chain for model in pdb_models for chain in model]

    if atom_level:
        atoms = list(set([a.get_name() for chain in chains for a in chain.get_atoms()]))
        encoder = {atom: i for i, atom in enumerate(atoms)}
        decoder = atoms
    else:
        residues = list(
            set([three_to_one[res.get_resname()] for chain in chains for res in chain])
        )
        encoder = {residue: i for i, residue in enumerate(residues)}
        decoder = residues

    return encoder, decoder


def get_coords_and_types(chain, encoder, atom_level=True, device="cpu"):
    """
    Get the coordinates and types of the chain
    :param chain: the chain
    :param encoder: the encoder
    :return: coords, types
    """
    if atom_level:
        coords = torch.tensor(
            np.array([a.get_coord() for a in chain.get_atoms()]),
            device=device,
            dtype=FLOAT_TYPE,
        )
        types = torch.tensor(
            [encoder[a.get_name()] for a in chain.get_atoms()],
            device=device,
        )
    else:
        coords = torch.tensor(
            np.array([res["CA"].get_coord() for res in chain]),
            device=device,
            dtype=FLOAT_TYPE,
        )
        types = torch.tensor(
            [encoder[three_to_one[res.get_resname()]] for res in chain],
            device=device,
        )

    return coords, types


def process_pmhc_pdb_file(
    pdb_path,
    encoder,
    atom_level=True,
    n_samples=1,
    device="cpu",
):
    """
    Process the pdb file to get the peptide and pocket representations
    :param pdb_path: the path of the pdb file
    :param pocket_atoms: whether to use atoms or residues for the pocket
    :param peptide_atoms: whether to use atoms or residues for the peptide
    :param n_samples: the batch size

    :return: peptide, mhc
    """
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_path)

    assert len(pdb_models) == 1
    pdb_model = pdb_models[0]

    mhc = pdb_model["M"]
    peptide = pdb_model["P"]

    mhc_coords, mhc_types = get_coords_and_types(
        mhc, encoder, atom_level=atom_level, device=device
    )
    peptide_coords, peptide_types = get_coords_and_types(
        peptide, encoder, atom_level=atom_level, device=device
    )

    mhc_one_hot = F.one_hot(mhc_types, num_classes=len(encoder))
    mhc_one_hot = torch.cat([mhc_one_hot] * n_samples, dim=0)
    mhc_size = torch.tensor(
        [len(mhc_coords)] * n_samples, device=device, dtype=INT_TYPE
    )
    mhc_mask = torch.repeat_interleave(
        torch.arange(n_samples, device=device, dtype=INT_TYPE), len(mhc_coords)
    )
    mhc_coords = torch.cat([mhc_coords] * n_samples, dim=0)

    mhc = {"x": mhc_coords, "one_hot": mhc_one_hot, "size": mhc_size, "mask": mhc_mask}

    peptide_one_hot = F.one_hot(peptide_types, num_classes=len(encoder))
    peptide_one_hot = torch.cat([peptide_one_hot] * n_samples, dim=0)

    peptide_size = torch.tensor(
        [len(peptide_coords)] * n_samples, device=device, dtype=INT_TYPE
    )
    peptide_mask = torch.repeat_interleave(
        torch.arange(n_samples, device=device, dtype=INT_TYPE), len(peptide_coords)
    )
    peptide_coords = torch.cat([peptide_coords] * n_samples, dim=0)
    peptide = {
        "x": peptide_coords,
        "one_hot": peptide_one_hot,
        "size": peptide_size,
        "mask": peptide_mask,
    }

    return peptide, mhc


def process_pmhc_directory(dir_path, **kwargs):
    """
    Process the directory to get the peptide and pocket representations
    :param dir_path: the path of the directory
    :param kwargs: the arguments to pass to process_pmhc_pdb_file
    :return: peptide, mhc
    """
    encoder, decoder = get_encoder_decoder(dir_path, atom_level=kwargs["atom_level"])
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
            name.append(pdb_path.stem)
            peptide_, mhc_ = process_pmhc_pdb_file(
                Path(dir_path) / pdb_file, encoder, **kwargs
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
    pdb__dir_path,
    outdir,
    **kwargs,
):
    peptide, mhc, encoder, decoder, names = process_pmhc_directory(
        pdb_path,
        **kwargs,
    )
    with json.open(Path(outdir) / "encoder.json", "w") as f:
        json.dump(encoder, f)

    with json.open(Path(outdir) / "decoder.json", "w") as f:
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

    n_nodes = get_n_nodes(peptide_mask, mhc_mask, smooth_sigma=1.0)
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


def write_updated_peptide_coords_pdb(
    peptide, decoder, pdb_reference_path, pdb_output_path, atom_level=True
):
    """
    Takes an existing pdb file with peptide and mhc and creates a new one
    with the same mhc pocket and the peptide with updated atom/residue
    coordinates given by the model.
    :param peptide: peptide with updated coordinates
    :param decoder: decoder, from index to atom/residue
    :param pdb_reference_path: path to the reference pdb file
    :param pdb_output_path: path to the output pdb file
    :param atom_level: whether to use atoms or residues

    :return: None
    """
    # Read the reference pdb file
    parser = PDBParser()
    structure = parser.get_structure("reference", pdb_reference_path)

    # Get the peptide chain
    peptide_chain = structure[0]["P"]

    # Get the peptide atoms
    peptide_atoms = peptide_chain.get_atoms()

    # Get the peptide residues
    peptide_residues = peptide_chain.get_residues()

    # Get the peptide atoms/residues
    if atom_level:
        peptide_elements = peptide_atoms
    else:
        peptide_elements = peptide_residues

    # Update the peptide coordinates
    for i, element in enumerate(peptide_elements):
        if atom_level:
            element.set_coord(peptide[i])
        else:
            for atom in element:
                atom.set_coord(peptide[i])

    # Write the new pdb file
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_output_path))
