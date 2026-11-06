# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens-compatible operation categories (no TraceLens dependency)."""

from __future__ import annotations

GEMM_OPS: frozenset[str] = frozenset({
    "aten::mm", "aten::addmm", "aten::_scaled_mm", "aten::bmm",
    "aten::baddbmm", "tex_ts::te_gemm_ts",
    "trtllm::cublas_scaled_mm", "bitsandbytes::int8_linear_matmul",
    "vllm::gemm_with_dynamic_quant",
    "_Linear_yfwd_mm", "_LinearBackward_xgrad_mm", "_LinearBackward_wgrad_mm",
    "_LayerNormLinear_yfwd_mm", "_LayerNormLinearBackward_xgrad_mm",
    "_LayerNormLinearBackward_wgrad_mm",
})

CONV_FWD_OPS: frozenset[str] = frozenset({
    "aten::convolution", "aten::miopen_convolution", "aten::cudnn_convolution",
    "ConvBias_", "ConvBiasReLU_",
})

CONV_BWD_OPS: frozenset[str] = frozenset({
    "aten::convolution_backward", "ConvBias_Backward", "ConvBiasReLU_Backward",
})

NORM_FWD_OPS: frozenset[str] = frozenset({
    "aten::batch_norm", "aten::native_batch_norm",
    "aten::miopen_batch_norm", "aten::cudnn_batch_norm",
    "aten::layer_norm", "aten::native_layer_norm",
    "aten::miopen_layer_norm", "aten::cudnn_layer_norm",
    "aten::group_norm", "aten::native_group_norm",
    "aten::miopen_group_norm", "aten::cudnn_group_norm",
    "aten::instance_norm", "aten::native_instance_norm",
    "aten::miopen_instance_norm", "aten::cudnn_instance_norm",
    "aten::_fused_rms_norm",
})

NORM_BWD_OPS: frozenset[str] = frozenset({
    "aten::batch_norm_backward", "aten::native_batch_norm_backward",
    "aten::miopen_batch_norm_backward", "aten::cudnn_batch_norm_backward",
    "aten::layer_norm_backward", "aten::native_layer_norm_backward",
    "aten::miopen_layer_norm_backward", "aten::cudnn_layer_norm_backward",
    "aten::group_norm_backward", "aten::native_group_norm_backward",
    "aten::miopen_group_norm_backward", "aten::cudnn_group_norm_backward",
    "aten::instance_norm_backward", "aten::native_instance_norm_backward",
    "aten::miopen_instance_norm_backward", "aten::cudnn_instance_norm_backward",
    "aten::rms_norm_backward", "aten::native_rms_norm_backward",
    "aten::miopen_rms_norm_backward", "aten::cudnn_rms_norm_backward",
    "aten::_fused_rms_norm_backward",
})

ALL_NORM_OPS: frozenset[str] = NORM_FWD_OPS | NORM_BWD_OPS

SDPA_FWD_OPS: frozenset[str] = frozenset({
    "FlashAttnFunc",
    "flash_attn::_flash_attn_forward", "flash_attn::_flash_attn_varlen_forward",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aiter::_flash_attn_forward", "aiter::wrapper_fmha_v3_fwd",
    "flash_attn_3::fwd",
})

SDPA_BWD_OPS: frozenset[str] = frozenset({
    "FlashAttnFuncBackward",
    "flash_attn::_flash_attn_backward", "flash_attn::_flash_attn_varlen_backward",
    "aten::_scaled_dot_product_cudnn_attention_backward",
    "aten::_scaled_dot_product_efficient_attention_backward",
    "aten::_scaled_dot_product_flash_attention_backward",
    "aiter::_flash_attn_backward", "aiter::wrapper_fmha_v3_bwd",
})

UNARY_ELEMENTWISE_OPS: frozenset[str] = frozenset({
    "aten::copy", "aten::copy_",
    "aten::clamp_min", "aten::clamp_min_",
    "aten::clamp_max", "aten::clamp_max_",
    "aten::sigmoid",
})

ACTIVATION_OPS: frozenset[str] = frozenset({
    "aten::gelu", "aten::relu", "aten::relu_", "aten::silu",
    "aten::tanh", "aten::neg", "aten::abs", "aten::exp",
})

BINARY_ELEMENTWISE_OPS: frozenset[str] = frozenset({
    "aten::div", "aten::div_",
    "aten::mul", "aten::mul_",
    "aten::add", "aten::add_",
    "aten::sigmoid_backward", "aten::threshold_backward",
})

REDUCE_OPS: frozenset[str] = frozenset({
    "aten::mean", "aten::sum", "aten::max", "aten::min",
    "aten::amax", "aten::amin", "aten::prod",
    "aten::norm", "aten::var", "aten::std",
})

SOFTMAX_OPS: frozenset[str] = frozenset({
    "aten::softmax", "aten::_softmax", "aten::log_softmax",
    "aten::_log_softmax",
})


def categorize_op(name: str) -> str:
    """Map a CPU op name to a raw TraceLens category."""
    if name in GEMM_OPS:
        return "GEMM"
    if name in CONV_FWD_OPS:
        return "CONV_fwd"
    if name in CONV_BWD_OPS:
        return "CONV_bwd"
    if name in ALL_NORM_OPS:
        return "NORM_bwd" if name.endswith("_backward") else "NORM_fwd"
    if name in SDPA_FWD_OPS:
        return "SDPA_fwd"
    if name in SDPA_BWD_OPS:
        return "SDPA_bwd"
    if name in SOFTMAX_OPS:
        return "other"
    if name in REDUCE_OPS:
        return "reduce"
    if name in ACTIVATION_OPS:
        return "elementwise"
    if name in UNARY_ELEMENTWISE_OPS:
        return "UnaryElementwise"
    if name in BINARY_ELEMENTWISE_OPS:
        return "BinaryElementwise"
    if name.startswith("triton"):
        return "triton"
    if name.startswith("record_param_comms"):
        return "record_param_comms"
    return "other"


_ENHANCED_NORM_SUBSTRINGS = (
    "batch_norm", "batchnorm", "layer_norm", "layernorm",
    "group_norm", "groupnorm", "instance_norm", "rms_norm",
)

_ENHANCED_MOE_SUBSTRINGS = ("moe", "fused_moe")


def display_category(op_name: str, raw_category: str) -> tuple[str, str]:
    """Return ``(slug, display_name)`` as TraceLens ``get_enhanced_category`` does."""
    lower = op_name.lower()
    if any(s in lower for s in _ENHANCED_MOE_SUBSTRINGS):
        return "moe_fused", "MoE Fused"
    if any(s in lower for s in _ENHANCED_NORM_SUBSTRINGS):
        return "norm", "Norm"
    if "conv" in lower and ("aten::" in op_name or "backward" in lower):
        return "convolution", "Convolution"

    slug = raw_category.replace(" ", "_").replace("/", "_").lower()
    display = _RAW_CATEGORY_DISPLAY.get(raw_category, raw_category)
    return slug, display


_RAW_CATEGORY_DISPLAY: dict[str, str] = {
    "GEMM": "GEMM",
    "CONV_fwd": "Convolution",
    "CONV_bwd": "Convolution (bwd)",
    "NORM_fwd": "Norm",
    "NORM_bwd": "Norm (bwd)",
    "SDPA_fwd": "SDPA",
    "SDPA_bwd": "SDPA (bwd)",
    "UnaryElementwise": "UnaryElementwise",
    "BinaryElementwise": "BinaryElementwise",
    "elementwise": "Elementwise",
    "reduce": "Reduce",
    "triton": "Triton",
    "other": "Other",
}


ALL_CATEGORIZED_OPS: frozenset[str] = (
    GEMM_OPS | CONV_FWD_OPS | CONV_BWD_OPS | ALL_NORM_OPS
    | SDPA_FWD_OPS | SDPA_BWD_OPS
    | UNARY_ELEMENTWISE_OPS | BINARY_ELEMENTWISE_OPS
    | ACTIVATION_OPS | REDUCE_OPS | SOFTMAX_OPS
)
