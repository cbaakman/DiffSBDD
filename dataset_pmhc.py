import io
import json
import os
from pathlib import Path
import warnings

from Bio.PDB import PDBParser, PDBIO
#from Bio.PDB.Polypeptide import three_to_one
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from pdb2sql import pdb2sql
from pdb2sql import interface as extract_interface
from Bio import BiopythonDeprecationWarning

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

INT_TYPE = torch.int64
FLOAT_TYPE = torch.float32


def get_encoder_decoder(all_types):
    """
    Get the encoder and decoder
    :param all_types: the types
    :return: encoder, decoder
    """
    elements = list(set(all_types))
    encoder = {element: i for i, element in enumerate(elements)}
    decoder = elements

    return encoder, decoder


def encode_types(types, encoder, device="cpu"):
    """
    Encode the types
    :param types: the types
    :param encoder: the encoder
    :return: the one-hot encoded types
    """
    encoded_types = [encoder[t] for t in types]
    one_hot_types = F.one_hot(
        torch.tensor(encoded_types, dtype=INT_TYPE, device=device), len(encoder)
    )
    return one_hot_types


def get_coords_and_types(chain, atom_level=True, device="cpu", interface_residue_ids=None):
    """
    Get the coordinates and types of the chain
    :param chain: the chain
    :return: coords, types
    """
    if atom_level:
        coords = torch.tensor(
            np.array([a.get_coord() for a in chain.get_atoms()]),
            device=device,
            dtype=FLOAT_TYPE,
        )
        types = [a.get_name() for a in chain.get_atoms()]
    else:
        coord_list = []
        types = []
        for res in chain:
            if interface_residue_ids is not None and res.id[1] not in interface_residue_ids:
                continue
            if "CA" in res:
                coord_list.append(res["CA"].get_coord())
                types.append(protein_letters_3to1[res.get_resname().capitalize()])
            else:
                print(f"Carbon alpha atom missing in residue {res}, chain {chain}")

        coords = torch.tensor(
            coord_list,
            device=device,
            dtype=FLOAT_TYPE,
        )

    return coords, types


def process_pmhc_pdb_file(
    pdb_path_or_stream,
    # encoder,
    atom_level=True,
    # n_samples=1,
    device="cpu",
    select_interface=False,
    pdb_string=None
):
    """
    Process the pdb file to get the peptide and pocket representations
    :param pdb_path: the path of the pdb file
    :param pocket_atoms: whether to use atoms or residues for the pocket
    :param peptide_atoms: whether to use atoms or residues for the peptide

    :return: peptide, mhc
    """
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_path_or_stream)

    assert len(pdb_models) == 1
    pdb_model = pdb_models[0]

    mhc_interface_residue_ids = None
    if select_interface:
        if pdb_string is None:
            pdb_db = pdb2sql(pdb_path_or_stream)
        else:
            pdb_db = pdb2sql(pdb_string)

        interface = extract_interface(pdb_db)
        interface_residues = interface.get_contact_residues(chain1="M", chain2="P")
        mhc_interface_residue_ids = [residue[1] for residue in interface_residues["M"]]

    mhc = pdb_model["M"]
    peptide = pdb_model["P"]

    mhc_coords, mhc_types = get_coords_and_types(
        mhc, atom_level=atom_level, device=device, interface_residue_ids=mhc_interface_residue_ids
    )
    peptide_coords, peptide_types = get_coords_and_types(
        peptide, atom_level=atom_level, device=device,
    )

    mhc_size = len(mhc_coords)
    mhc = {"x": mhc_coords, "types": mhc_types, "size": mhc_size}


    peptide_size = len(peptide_coords)
    peptide = {
        "x": peptide_coords,
        "types": peptide_types,
        "size": peptide_size,
    }

    return peptide, mhc


def process_pmhc_hdf5_file(hdf5_path, device="cpu", atom_level=True, select_interface=False):
    """
    Process the hdf5 file to get the peptide and pocket representations
    :param hdf5_path: the path of the hdf5 file
    :return: peptides, mhcs
    """

    pdb_strings, names = read_pdb_strings_hdf5_file(hdf5_path)

    peptides = []
    mhcs = []
    for pdb_string in pdb_strings:
        pdb_stream = io.StringIO(pdb_string)
        peptide, mhc = process_pmhc_pdb_file(
            pdb_stream, device=device, atom_level=atom_level, select_interface=select_interface, pdb_string=pdb_string
        )

        peptides.append(peptide)
        mhcs.append(mhc)

    return peptides, mhcs, names


def read_pdb_strings_hdf5_file(hdf5_path):
    """
    Read the pdb string from the hdf5 file
    :param hdf5_path: the path of the hdf5 file
    :return: the pdb string
    """
    content = h5py.File(hdf5_path, "r")
    pdb_strings = []
    names = []
    for name, model in content.items():
        pdb_string = "\n".join([line.decode("utf-8") for line in model["complex"]])
        pdb_strings.append(pdb_string)
        names.append(name)

    return pdb_strings, names


def write_updated_peptide_coords_pdb(
    peptide, decoder, pdb_reference_path_or_stream, pdb_output_path, atom_level=True
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
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_reference_path_or_stream)

    # Get the peptide chain
    peptide_chain = pdb_models[0]["P"]

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
        pdb_models[0].detach_child("P")
        pdb_models[0].add(peptide_chain_new)

        # Write the new pdb file
    io = PDBIO()
    io.set_structure(pdb_models)
    io.save(str(pdb_output_path))
