# This file is deprecated.  Its contents have been migrated to:
#   archs/example_arch/genotypes/cifar.py   (search-space definition)
#   archs/example_arch/genotypes/__init__.py (get_search_space interface)
#
# The search space is now owned by each arch package and accessed via
# archs.load_arch(arch_name).get_search_space(dataset).
# This file is kept only to prevent ImportError in old scripts;
# it can be deleted once all callers are updated.

import warnings
warnings.warn(
    "ea.genotypes is deprecated. "
    "Use archs.load_arch(arch_name).get_search_space(dataset) instead.",
    DeprecationWarning,
    stacklevel=2,
)
