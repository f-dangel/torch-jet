"""Execute the README's quickstart example to keep it from going stale.

The README ships a runnable ``python`` code block whose final ``assert``
verifies the computed third-order derivative against a hand-written value.
This test extracts that block and ``exec``s it, so a broken README reddens CI.
"""

from pathlib import Path

#: Repository root (two levels up from this file: ``test/`` then repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: HTML markers delimiting the quickstart code block in ``README.md``.
BEGIN_MARKER = "<!-- BEGIN quickstart -->"
END_MARKER = "<!-- END quickstart -->"


def extract_quickstart() -> str:
    """Extract the python code block from the README's quickstart section.

    Returns:
        The source code inside the first ```python fence located between the
        ``BEGIN_MARKER`` and ``END_MARKER`` HTML comments.

    Raises:
        ValueError: If the markers or the fenced ```python block are missing.
    """
    readme = (REPO_ROOT / "README.md").read_text()

    try:
        section = readme.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    except IndexError as e:
        raise ValueError("Could not locate quickstart markers in README.md") from e

    if "```python" not in section:
        raise ValueError("No ```python block found in the quickstart section")
    return section.split("```python", 1)[1].split("```", 1)[0]


def test_readme_quickstart_runs():
    """The README quickstart executes top-to-bottom and its assert holds."""
    code = extract_quickstart()
    exec(compile(code, "README.md::quickstart", "exec"), {})
