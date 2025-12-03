# Package marker
 
"""Top-level package exports for `missingfcup`.

This module exposes the public API of the package. Importing
`missingfcup_pkg` will make `missing_matrix` available at the
package level.

Example:
	from missingfcup import missing_matrix

"""

# Import the primary function so it becomes available as
# `missingfcup.missing_matrix` for consumers of the package.
from .viz.missing_matrix import missing_matrix

# Specify the public symbols exported by `from missingfcup import *`.
__all__ = ["missing_matrix"]