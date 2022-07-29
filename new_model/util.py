# Copyright 2021 Jacob Durrant

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


'''
rdkit/openbabel utility scripts
'''
import numpy as np


# try:
#     import pybel
# except:
#     from openbabel import pybel

from rdkit import Chem


def get_coords(mol):
    """Returns an array of atom coordinates from an rdkit mol."""
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(conf.GetNumAtoms())])
    return coords


def get_types(mol):
    """Returns an array of atomic numbers from an rdkit mol."""
    return [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]


def combine_all(frags):
    """Combines a list of rdkit mols."""
    if len(frags) == 0:
        return None

    c = frags[0]
    for f in frags[1:]:
        c = Chem.CombineMols(c,f)

    return c


def generate_fragments(mol, max_heavy_atoms=0, only_single_bonds=True):
    """Takes an rdkit molecule and returns a list of (parent, fragment) tuples.

    Args:
        mol: The molecule to fragment.
        max_heavy_atoms: The maximum number of heavy atoms to include
            in generated fragments.
        nly_single_bonds: If set to true, this method will only return
            fragments generated by breaking single bonds.

    Returns:
        A list of (parent, fragment) tuples where mol is larger than fragment.
    """
    # list of (parent, fragment) tuples
    splits = []

    # if we have multiple ligands already, split into pieces and then iterate
    ligands = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i in range(len(ligands)):
        lig = ligands[i]
        other = list(ligands[:i] + ligands[i+1:])

        # iterate over bonds
        for i in range(lig.GetNumBonds()):
            # (optional) filter base on single bonds
            if only_single_bonds and lig.GetBondWithIdx(i).GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue

            # split the molecule
            split_mol = Chem.rdmolops.FragmentOnBonds(lig, [i])

            # obtain fragments
            fragments = Chem.GetMolFrags(split_mol, asMols=True, sanitizeFrags=False)

            # skip if this did not break the molecule into two pieces
            if len(fragments) != 2:
                continue

            # otherwise make sure the first fragment is larger
            if fragments[0].GetNumAtoms() < fragments[1].GetNumAtoms():
                fragments = fragments[::-1]

            # make sure the fragment has at least one heavy atom
            if fragments[1].GetNumHeavyAtoms() == 0:
                continue

            # (optional) filter based on number of heavy atoms in the fragment
            if max_heavy_atoms > 0 and fragments[1].GetNumHeavyAtoms() > max_heavy_atoms:
                continue

            # if we have other ligands present, merge them with the parent
            parent = fragments[0]

            if len(other) > 0:
                parent = combine_all([parent] + other)

            # add this pair
            splits.append((parent, fragments[1]))

    return splits


def load_ligand(sdf):
    """Loads a ligand from an sdf file and fragments it.

    Args:
        sdf: Path to sdf file containing a ligand.
    """
    lig = next(Chem.SDMolSupplier(sdf, sanitize=False))
    frags = generate_fragments(lig)

    return lig, frags


def load_ligands_pdb(pdb):
    """Load multiple ligands from a pdb file.

    Args:
        pdb: Path to pdb file containing a ligand.
    """
    lig_mult = Chem.MolFromPDBFile(pdb)
    ligands = Chem.GetMolFrags(lig_mult, asMols=True, sanitizeFrags=True)

    return ligands


def remove_water(m):
    """Removes water molecules from an rdkit mol."""
    parts = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    valid = [k for k in parts if not Chem.MolToSmiles(k, allHsExplicit=True) == '[OH2]']

    assert len(valid) > 0, 'error: molecule contains only water'

    merged = valid[0]
    for part in valid[1:]:
        merged = Chem.CombineMols(merged, part)

    return merged


def load_receptor(rec_path):
    """Loads a receptor from a pdb file and retrieves atomic information.

    Args:
        rec_path: Path to a pdb file.
    """
    rec = Chem.MolFromPDBFile(rec_path, sanitize=False)
    rec = remove_water(rec)

    return rec


# def load_receptor_ob(rec_path):
#     rec = next(pybel.readfile('pdb', rec_path))
#     valid = [r for r in rec.residues if r.name != 'HOH']

#     # map partial charge into byte range
#     def conv_charge(x):
#         x = max(x,-0.5)
#         x = min(x,0.5)
#         x += 0.5
#         x *= 255
#         x = int(x)
#         return x

#     coords = []
#     types = []
#     for v in valid:
#         coords += [k.coords for k in v.atoms]
#         types += [(
#             k.atomicnum,
#             int(k.OBAtom.IsAromatic()),
#             int(k.OBAtom.IsHbondDonor()),
#             int(k.OBAtom.IsHbondAcceptor()),
#             conv_charge(k.OBAtom.GetPartialCharge())
#         ) for k in v.atoms]

#     return np.array(coords), np.array(types)


def load_receptor_ob(rec_path):
    rec = load_receptor(rec_path)

    coords = get_coords(rec)
    types = np.array(get_types(rec))
    types = np.concatenate([
        types.reshape(-1,1), 
        np.zeros((len(types), 4))
    ], 1)

    return coords, types


def get_connection_point(frag):
    '''return the coordinates of the dummy atom as a numpy array [x,y,z]'''
    dummy_idx = get_types(frag).index(0)
    coords = get_coords(frag)[dummy_idx]

    return coords


def frag_dist_to_receptor(rec, frag):
    '''compute the minimum distance between the fragment connection point any receptor atom'''
    rec_coords = rec.GetConformer().GetPositions()
    conn = get_connection_point(frag)

    dist = np.sum((rec_coords - conn) ** 2, axis=1)
    min_dist = np.sqrt(np.min(dist))

    return min_dist


def frag_dist_to_receptor_raw(coords, frag):
    '''compute the minimum distance between the fragment connection point any receptor atom'''
    rec_coords = np.array(coords)
    conn = get_connection_point(frag)

    dist = np.sum((rec_coords - conn) ** 2, axis=1)
    min_dist = np.sqrt(np.min(dist))

    return min_dist


def mol_array(mol):
    '''convert an rdkit mol to an array of coordinates and atom types'''
    coords = get_coords(mol)
    types = np.array(get_types(mol)).reshape(-1,1)

    arr = np.concatenate([coords, types], axis=1)

    return arr


def desc_mol_array(mol, atom_fn):
    '''user-defined atomic mapping function'''
    coords = get_coords(mol)
    atoms = list(mol.GetAtoms())
    types = np.array([atom_fn(x) for x in atoms]).reshape(-1,1)

    arr = np.concatenate([coords, types], axis=1)

    return arr


def desc_mol_array_ob(atoms, atom_fn):
    coords = np.array([k[0] for k in atoms])
    types = np.array([atom_fn(k[1]) for k in atoms]).reshape(-1,1)

    # arr = np.concatenate([coords, types], axis=1)

    return coords, types


def mol_to_points(mol, atom_types=[6,7,8,9,15,16,17,35,53]):
    '''convert an rdkit mol to an array of coordinates and layers'''
    coords = get_coords(mol)

    types = get_types(mol)
    layers = np.array([(atom_types.index(k) if k in atom_types else -1) for k in types])

    # filter by existing layer
    coords = coords[layers != -1]
    layers = layers[layers != -1].reshape(-1,1)

    return coords, layers


def merge_smiles(sma, smb):
    '''merge two smile frament strings by combining at the dummy connection point'''
    a = Chem.MolFromSmiles(sma, sanitize=False)
    b = Chem.MolFromSmiles(smb, sanitize=False)

    # merge molecules
    c = Chem.CombineMols(a,b)

    # find dummy atoms
    da,db = np.where(np.array([k.GetAtomicNum() for k in c.GetAtoms()]) == 0)[0]

    # find neighbors to connect
    na = c.GetAtomWithIdx(int(da)).GetNeighbors()[0].GetIdx()
    nb = c.GetAtomWithIdx(int(db)).GetNeighbors()[0].GetIdx()

    e = Chem.EditableMol(c)
    for d in sorted([da,db])[::-1]:
        e.RemoveAtom(int(d))

    # adjust atom indexes
    na -= int(da < na) + int(db < na)
    nb -= int(da < nb) + int(db < nb)

    e.AddBond(na,nb,Chem.rdchem.BondType.SINGLE)

    r = e.GetMol()

    sm = Chem.MolToSmiles(Chem.RemoveHs(r, sanitize=False), isomericSmiles=False)

    return sm
