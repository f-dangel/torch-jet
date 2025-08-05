"""Configuration file for `mkdocs-gallery` export of examples."""

import os
import re
import shutil
from pathlib import Path

from mkdocs_gallery.gen_gallery import DefaultResetArgv
from mkdocs_gallery.sorting import FileNameSortKey

# See
# https://sphinx-gallery.github.io/stable/_modules/sphinx_gallery/gen_gallery.html
# for options
conf = {
    "reset_argv": DefaultResetArgv(),
    "filename_pattern": f"{re.escape(os.sep)}examples",
    "abort_on_example_error": True,
    # order examples according to file name
    "within_subsection_order": FileNameSortKey,
}


def copy_images():
    """Copy image files to the gallery output directory."""
    src_dir = Path("docs/examples")
    dst_dir = Path("docs/generated/gallery")
    # Create destination directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy all PNG files
    for img_file in src_dir.glob("*.png"):
        shutil.copy(img_file, dst_dir / img_file.name)


copy_images()
