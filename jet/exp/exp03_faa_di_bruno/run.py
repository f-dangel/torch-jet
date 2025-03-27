"""Produce LaTeX expressions for the Faà di Bruno formula."""

from typing import Any

from jet.utils import integer_partitions, multiplicity


def _subscript(arg: str, sub: Any) -> str:
    """Add a LaTeX subscript to an argument.

    Args:
        arg: The argument.
        sub: The subscript.

    Returns:
        The argument with the subscript.
    """
    return arg + "_{" + str(sub) + "}"


def _superscript(arg: str, sup: Any) -> str:
    """Add a LaTeX superscript to an argument.

    Args:
        arg: The argument.
        sup: The superscript.

    Returns:
        The argument with the subscript.
    """
    return arg + "^{" + str(sup) + "}"


def latex_faa_di_bruno(k: int, x: str = "x", h: str = "h") -> str:
    """Produce LaTeX expressions for the Faà di Bruno formula for h(x).

    Args:
        k: The order of the derivative.
        x: The variable. Defaults to "x".
        h: The function. Defaults to "h".

    Returns:
        A LaTeX expression for the k-th derivative in Faà di Bruno's formula.
    """
    vx = rf"\v{x}"
    vh = rf"\v{h}"

    equation = f"{_subscript(vh, k)} \n\t="

    if k == 0:
        return f"{equation}\n\t\t{h}({_subscript(vx, 0)})"

    partitions = list(integer_partitions(k))
    # sort by descending length of the partition
    partitions.sort(key=len, reverse=True)

    if len(partitions) > 1:
        equation += "\n\t" + r"\begin{matrix}"

    for idx, sigma in enumerate(partitions):
        if idx != 0:
            equation += "\n\t" + r"\\" + "\n\t+"

        vxs = ", ".join([_subscript(vx, i) for i in sigma])

        deriv = r"\partial" if k > 0 else ""
        if len(sigma) > 1:
            deriv = _superscript(deriv, len(sigma))

        nu = multiplicity(sigma)
        assert int(nu) == nu
        nu_str = "" if nu == 1.0 else f"{int(nu)} "
        term = f"{nu_str}{deriv} {h} [{vxs}]"

        equation += f"\n\t\t{term}"

    if len(partitions) > 1:
        equation += "\n\t" + r"\end{matrix}"

    return equation


def latex_faa_di_bruno_composition(
    k: int, x: str = "x", h: str = "h", g: str = "g", f: str = "f"
) -> str:
    """Produce LaTeX expressions for the Faà di Bruno formula for f(x) = (g ∘ h)(x).

    Args:
        k: The order of the derivative.
        x: The variable. Defaults to "x".
        h: The second composite. Defaults to "h".
        g: The first composite. Defaults to "g".
        f: The composite. Defaults to "f".

    Returns:
        A LaTeX expression for the k-th derivative in Faà di Bruno's formula.
    """
    vx = rf"\v{x}"

    equation = f"{_subscript(vx, k)}" + "\n\t" + r"\to" + "\n\t"
    equation += latex_faa_di_bruno(k, x=x, h=h).replace("\n", "\n\t")

    equation += "\n\t" + r"\to" + "\n\t\t"
    equation += latex_faa_di_bruno(k, x=h, h=g).replace("\n", "\n\t\t")

    equation += "\n\t=\n\t\t"
    equation += latex_faa_di_bruno(k, x=x, h=f).replace("\n", "\n\t\t")

    return equation


if __name__ == "__main__":
    K_max = 8

    # You can copy the output of this and put it into a
    # \begin{align*} ... \end{align} LaTeX environment.
    for k in range(K_max + 1):
        fdb = latex_faa_di_bruno_composition(k)
        # post-process for prettier formatting
        fdb = fdb.replace(r"\to", r"&\to&")
        fdb = fdb.replace(r"\vf", r"&\vf")
        print(fdb)
        if k != K_max:
            print(r"\\")
