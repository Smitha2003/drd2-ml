from rdkit import DataStructs


def compute_max_similarity(input_bv, training_bitvectors):
    """
    Compute maximum Tanimoto similarity between
    input fingerprint and training fingerprints.
    """
    similarities = [
        DataStructs.TanimotoSimilarity(input_bv, train_bv)
        for train_bv in training_bitvectors
    ]

    return max(similarities)