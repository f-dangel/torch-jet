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


def _tensor_outer(arg: str, power: int) -> str:
    """Add a LaTeX tensor outer product to an argument if the power exceeds 1.

    Args:
        arg: The argument.
        power: The power.

    Returns:
        The argument with the tensor outer product.
    """
    return arg if power == 1 else arg + r"^{\otimes" + str(power) + r"}"


def latex_faa_di_bruno(k: int, x: str = r"\vx", h: str = r"\vh") -> str:
    """Produce LaTeX expressions for the Faà di Bruno formula for h(x).

    Args:
        k: The order of the derivative.
        x: The variable. Defaults to "x".
        h: The function. Defaults to "h".

    Returns:
        A LaTeX expression for the k-th derivative in Faà di Bruno's formula.
    """
    equation = f"{_subscript(h, k)} \n\t="

    if k == 0:
        return f"{equation}\n\t\t{h}({_subscript(x, 0)})"

    partitions = list(integer_partitions(k))
    # sort by descending length of the partition
    partitions.sort(key=len, reverse=True)

    if len(partitions) > 1:
        equation += "\n\t" + r"\begin{matrix}"

    for idx, sigma in enumerate(partitions):
        if idx != 0:
            equation += "\n\t" + r"\\" + "\n\t+"

        counts = {s: sigma.count(s) for s in set(sigma)}
        vxs = r"\otimes ".join(
            [_tensor_outer(_subscript(x, i), power) for i, power in counts.items()]
        )

        deriv = r"\partial" if k > 0 else ""
        if len(sigma) > 1:
            deriv = _superscript(deriv, len(sigma))

        nu = multiplicity(sigma)
        assert int(nu) == nu
        nu_str = "" if nu == 1.0 else f"{int(nu)} "
        term = f"{nu_str}" + r"\langle " + f"{deriv} {h}, {vxs}" + r" \rangle"

        equation += f"\n\t\t{term}"

    if len(partitions) > 1:
        equation += "\n\t" + r"\end{matrix}"

    return equation


def latex_faa_di_bruno_composition(
    k: int, x: str = r"\vx", h: str = r"\vh", g: str = r"\vg", f: str = r"\vf"
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
    equation = f"{_subscript(x, k)}" + "\n\t" + r"\to" + "\n\t"
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
        fdb = fdb.replace(r"\vf_", r"&\vf_")
        print(fdb)
        if k != K_max:
            print(r"\\")
