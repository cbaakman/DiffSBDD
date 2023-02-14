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
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

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
            set([three_to_one(res.get_resname()) for chain in chains for res in chain])
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
            [encoder[three_to_one(res.get_resname())] for res in chain],
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

    if not atom_level:
        peptide_chain_new = Chain("P")

    # Get the peptide atoms/residues
    if atom_level:
        peptide_elements = peptide_chain.get_atoms()
    else:
        peptide_elements = peptide_chain.get_residues()

    # Update the peptide coordinates
    for i, element in enumerate(peptide_elements):
        if atom_level:
            element.set_coord(peptide[i])
        else:
            ca_atom = element["CA"]
            ca_atom.set_coord(peptide[i])
            new_residue = Residue(element.get_id(), element.get_resname(), "")
            new_residue.add(ca_atom)
            peptide_chain_new.add(new_residue)

    # Write the new pdb file
    if not atom_level:
        structure[0].detach_child("P")
        structure[0].add(peptide_chain_new)

        # Write the new pdb file
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_output_path))
