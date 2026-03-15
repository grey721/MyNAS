"""
example_arch net sub-package.

Re-exports all symbols that ``archs/example_arch/__init__.py`` (and thus
``archs.load_arch()``) needs from this sub-package.

Public symbols
--------------
``Net``                  Required.  Runtime chromosome decoder.
``initialize_population`` Required.  Population factory for Searcher.
``generate_code``         Optional.  Static code generator; enables static
                          save mode in ResultSaver.
``sample_gene``           Optional.  Per-block mutation primitive for ga.py.
``NUM_GENE_PER_BLOCK``    Informational.  Gene vector length (= 4).
"""

from archs.example_arch.net.codegen               import generate_code
from archs.example_arch.net.net                   import Net
from archs.example_arch.net.population_initializer import (
    NUM_GENE_PER_BLOCK,
    initialize_population,
    sample_gene,
)

__all__ = [
    "Net",
    "initialize_population",
    "sample_gene",
    "generate_code",
    "NUM_GENE_PER_BLOCK",
]
