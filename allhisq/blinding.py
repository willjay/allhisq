"""
Module for generating, writing, and reading opaque blinding factors
"""
import hashlib
from array import array
import numpy as np


def main():
    """
    Generates a blinding factor, saves it to disk, and verifies that reading was successful.
    """
    seed_str = 'allhisq_blind_v12345'
    fname = f"{seed_str}.bin"
    blind = compute_blinding_factor(seed_str)
    write_blind(fname, blind)
    blind2 = read_blind(fname)

    print("Created blind =", blind)
    print("Read blind from ", fname)
    print("Read blind =", blind2)
    print("Blinds matched before and after reading?", blind == blind2)


class Blind(float):
    """
    An "opaque" version of float which hides its value in print statements but can be used normally
    in arithematic.
    """
    def __new__(cls, value):
        return super().__new__(cls, value)

    def __init__(self, value):
        float.__init__(value)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Blind(<opaque value>)"


def seed_from_string(seed_str):
    """
    Creates a deterministically random seed from an input string using hashing.
    Credit: StackExchange user jakevdp
    https://stackoverflow.com/questions/36755940/seed-numpy-random-randomstate-with-hashlib-hash
    Args:
        seed_str: str, the string to hash and convert to the seed
    Returns:
        seed: array of integers to be used with np.random.seed(seed)
    """
    assert isinstance(seed_str, str)
    ahash = hashlib.sha256(seed_str.encode('utf-8'))
    seed = np.frombuffer(ahash.digest(), dtype='uint32')
    return seed


def compute_blinding_factor(seed_str):
    """
    Computes a deterministically random blinding factor between 0.95 and 1.05 given an input string,
    which is hashed to seed the random number generator.
    Args:
        seed_str: str, the string to hash and use as the seed
    Returns:
        blind: Blind between 0.95 and 1.05
    """
    seed = seed_from_string(seed_str)
    np.random.seed(seed)
    return Blind(np.random.uniform(0.95, 1.05))


def write_blind(fname, blind):
    """
    Writes the blinding factor (in binary) to file.
    Args:
        fname: str, the full path to the file
        blind: float, the blinding factor
    """
    with open(fname, 'wb') as ofile:
        float_array = array('d', [blind])
        float_array.tofile(ofile)


def read_blind(fname):
    """
    Reads the blinding factor from file.
    Args:
        fname: str, the full path to the file
    Returns:
        Blind
    """
    with open(fname, 'rb') as ifile:
        blind = array('d')
        blind.frombytes(ifile.read())
        blind, = blind
    return Blind(blind)


if __name__ == '__main__':
    main()
