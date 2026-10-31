"""Jets of the vector-Jacobian product of every primitive case.

A jet of a function that differentiates inside traces to a graph of autograd's
backward ops, so this file checks the jet rules for those ops. The cases whose
backward op has no rule yet are strict expected failures on one cell of the
grid: a case that starts passing must move into ``SUPPORTED``.
"""

from typing import Any

from pytest import FixtureRequest, mark

from test.test_primitives import PRIMITIVE_CASES
from test.utils import assert_jet_matches_oracle, vjp_case

#: Primitive cases whose VJP has a jet rule for every backward op it needs.
SUPPORTED = set(
    """
sin-1d sin-2d cos-1d cos-2d tanh-1d tanh-2d exp-1d exp-2d erf-1d erf-2d log-1d
log-2d rsqrt-1d rsqrt-2d pow_float pow_float_int pow_int_0 pow_int_5 pow_int_10
pow_int_negative add_JJ add_JJ_aliased add_JC add_CJ add_scalar-1d add_scalar-2d
pad_value pad_crop pad_2d mul_scalar sub_JJ sub_JC sub_CJ mul_JJ mul_JJ_aliased
mul_JJ_diff_rank mul_JC mul_CJ add_1x4_3x1_JJ add_1x4_3x1_JC add_1x4_3x1_CJ
add_1x1_3x4_JJ add_1x1_3x4_JC add_1x1_3x4_CJ add_4_3x4_JJ add_4_3x4_JC add_4_3x4_CJ
add_3x4_4_JJ add_3x4_4_JC add_3x4_4_CJ add_4_3x1_JJ add_4_3x1_JC add_4_3x1_CJ
add_3x1_4_JJ add_3x1_4_JC add_3x1_4_CJ add_2x1x4_1x3x1_JJ add_2x1x4_1x3x1_JC
add_2x1x4_1x3x1_CJ add_2x1x4_3x1_JJ add_2x1x4_3x1_JC add_2x1x4_3x1_CJ sub_1x4_3x1_JJ
sub_1x4_3x1_JC sub_1x4_3x1_CJ sub_1x1_3x4_JJ sub_1x1_3x4_JC sub_1x1_3x4_CJ
sub_4_3x4_JJ sub_4_3x4_JC sub_4_3x4_CJ sub_3x4_4_JJ sub_3x4_4_JC sub_3x4_4_CJ
sub_4_3x1_JJ sub_4_3x1_JC sub_4_3x1_CJ sub_3x1_4_JJ sub_3x1_4_JC sub_3x1_4_CJ
sub_2x1x4_1x3x1_JJ sub_2x1x4_1x3x1_JC sub_2x1x4_1x3x1_CJ sub_2x1x4_3x1_JJ
sub_2x1x4_3x1_JC sub_2x1x4_3x1_CJ mul_1x4_3x1_JJ mul_1x4_3x1_JC mul_1x4_3x1_CJ
mul_1x1_3x4_JJ mul_1x1_3x4_JC mul_1x1_3x4_CJ mul_4_3x4_JJ mul_4_3x4_JC mul_4_3x4_CJ
mul_3x4_4_JJ mul_3x4_4_JC mul_3x4_4_CJ mul_4_3x1_JJ mul_4_3x1_JC mul_4_3x1_CJ
mul_3x1_4_JJ mul_3x1_4_JC mul_3x1_4_CJ mul_2x1x4_1x3x1_JJ mul_2x1x4_1x3x1_JC
mul_2x1x4_1x3x1_CJ mul_2x1x4_3x1_JJ mul_2x1x4_3x1_JC mul_2x1x4_3x1_CJ where_JJ
where_JC where_CJ where_scalar masked_fill_scalar masked_fill_tensor where_JJ_bcast
mm_CJ mm_JC mm_JJ addmm_CCJ addmm_CJC addmm_CJJ addmm_JCC addmm_JCC_bcast addmm_JCJ
addmm_JCJ_bcast addmm_JJC addmm_JJC_bcast addmm_JJJ addmm_JJJ_bcast bmm_CJ bmm_JC
bmm_JJ mean_all mean_dim mean_dim_keepdim cat_JJ cat_JJ_dim0 cat_JJ_dim_neg
cat_jet_const cat_jet_empty_const split chunk_shufflenet stack_JJ_dim0 stack_JJ_dim1
stack_JJ_dim-1 stack_jet_const rms_norm_inJ_wN_bN_1d rms_norm_inJ_wJ_bN_1d
rms_norm_inJ_wC_bN_1d rms_norm_inC_wJ_bN_1d rms_norm_inJ_wJ_bN_2d
rms_norm_inJ_wJ_bN_1d_default_eps sum_all sum_dim_0 sum_dim_list sum_keepdim
sum_dim_list_keepdim neg div_scalar div_tensor div_tensor_const div_JJ div_JJ_bcast
div_const_J t view _unsafe_view unsqueeze squeeze squeeze_dims transpose permute
expand repeat_interleave clone contiguous to_copy_f32 alias zeros_like
zeros_like_dtype_cast ones_like
""".split()
)
#: MPS has no fused attention kernel, so ``sdpa`` with constant query and key
#: differentiates through ``softmax(scores) @ value``, which needs no new rule.
SUPPORTED_ON = {
    "mps": {
        f"sdpa_CCJ{s}"
        for s in ("", "_causal", "_bool_mask", "_float_mask", "_scale", "_gqa")
    }
}
CASES = [vjp_case(c) for c in PRIMITIVE_CASES]


@mark.parametrize(
    "config", [c for c in CASES if c["id"] in SUPPORTED], ids=lambda c: c["id"]
)
def test_vjp(config: dict[str, Any], K: int, collapsed: bool, device: str):
    """``jet(vjp(primitive))`` matches its mode-specific oracle."""
    assert_jet_matches_oracle(config, K, collapsed, device)


@mark.parametrize(
    "config", [c for c in CASES if c["id"] not in SUPPORTED], ids=lambda c: c["id"]
)
def test_vjp_unsupported(config: dict[str, Any], device: str, request: FixtureRequest):
    """A VJP whose backward op has no jet rule raises ``NotImplementedError``."""
    if config["id"] not in SUPPORTED_ON.get(device, ()):
        request.applymarker(
            mark.xfail(strict=True, raises=NotImplementedError, reason="no jet rule")
        )
    assert_jet_matches_oracle(config, K=1, collapsed=False, device=device)
