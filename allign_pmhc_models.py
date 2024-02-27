import argparse
import json
from pathlib import Path

from Bio import PDB
from Bio.PDB import Superimposer

def align_pmhc(path_fixed: Path, path_moving: Path, save_path: Path) -> None:
    # Parse the PDB files
    structure_fixed = PDB.PDBParser().get_structure("setA", path_fixed)
    structure_moving = PDB.PDBParser().get_structure("setB", path_moving)
    # Extract the MHC chains and peptide chains from both sets
    MHC_chain_fixed = structure_fixed[0]["M"]
    MHC_chain_moving = structure_moving[0]["M"]
    
    # Create lists to store the coordinates of MHC and peptide residues from both sets
    MHC_coords_fixed = []
    MHC_coords_moving = []

    # Iterate over the MHC chains and store their coordinates for set A
    for residue in MHC_chain_fixed:
        if residue.get_id()[0] == " ":
            MHC_coords_fixed.append(residue["CA"])

    # Iterate over the MHC chains and store their coordinates for set B
    for residue in MHC_chain_moving:
        if residue.get_id()[0] == " ":
            MHC_coords_fixed.append(residue["CA"])

    # Create Superimposer object for MHC chains
    sup_MHC = Superimposer()
    sup_MHC.set_atoms(MHC_coords_fixed, MHC_coords_moving)
    sup_MHC.apply(structure_moving[0]["M"].get_atoms())
    sup_MHC.apply(structure_moving[0]["P"].get_atoms())
    
    # Save the aligned structure of set B to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(structure_moving)
    io.save(save_path)


def allign_directory_to_target(moving_dir: Path, fixed_path: Path, save_dir: Path, ids_to_keep=[]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for path in moving_dir.iterdir():
        if path.suffix == ".pdb":
            if ids_to_keep and path.stem not in ids_to_keep:
                continue
            
            save_path = save_dir / path.name
            align_pmhc(fixed_path, path, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving_dir", type=Path, required=True)
    parser.add_argument("--fixed_path", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--id_json_path", type=Path, required=False, default=None)
    args = parser.parse_args()

    if args.id_json_path is not None:
        with open(args.id_json_path, "r") as f:
            ids_to_keep = json.load(f)
    else:
        ids_to_keep = None

    allign_directory_to_target(args.moving_dir, args.fixed_path, args.save_dir, ids_to_keep=ids_to_keep)