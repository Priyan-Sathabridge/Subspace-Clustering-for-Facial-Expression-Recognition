"""
registry.py

Registry of available affinity matrix
construction methods.
"""

from .symmetric import SymmetricAffinity


AFFINITY_METHODS = {

    "symmetric": SymmetricAffinity,

}