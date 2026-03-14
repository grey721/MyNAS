"""
example_arch arch package entry point.

Exposes the unified interface required by ``archs.load_arch()``.

Directory layout
----------------
example_arch/
  genotypes/   Search-space definitions (candidate-value lists).
  modules/     Primitive operators / Block implementation.
  net/         Chromosome-to-network decoder (``Net``) + population initialiser.
"""

from archs.example_arch.genotypes import get_search_space
from archs.example_arch.net import Net, initialize_population, generate_code

__all__ = ['Net', 'initialize_population', 'get_search_space', 'generate_code']
