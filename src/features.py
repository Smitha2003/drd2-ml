import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def initialize_generator(radius: int, n_bits: int):
    """
    Initialize Morgan fingerprint generator.
    """
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits
    )


def smiles_to_fingerprint(smiles: str, generator, n_bits: int):
    """
    Convert SMILES to Morgan fingerprint numpy array.
    Returns (mol, fingerprint_array)
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return mol, arr


def numpy_to_bitvect(arr: np.ndarray):
    """
    Convert numpy fingerprint array to RDKit ExplicitBitVect.
    """
    bitstring = "".join(arr.astype(str))
    return DataStructs.CreateFromBitString(bitstring)