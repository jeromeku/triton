from collections import OrderedDict

MATMUL_ARGS = [
    "C",
    "A",
    "B",
    "M",
    "N",
    "K",
    "stride_cm",
    "stride_cn",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
]


MATMUL_CONSTANTS = ["BLOCK_M", "BLOCK_N", "BLOCK_K"]


DEFAULT_MATMUL_DTYPES = OrderedDict(
    {
        "C": "*fp32",
        "A": "*fp16",
        "B": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
    }
)
DEFAULT_MATMUL_HINTS = OrderedDict(
    {
        "C": 16,
        "A": 16,
        "B": 16,
        "M": None,
        "N": None,
        "K": None,
        "stride_cm": None,
        "stride_cn": 1,
        "stride_am": None,
        "stride_ak": 1,
        "stride_bk": 16,
        "stride_bn": 1,
    }
)


DEFAULT_MATMUL_CONSTANTS = OrderedDict({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16})


NO_HINTS = {k: None for k in MATMUL_ARGS}
STRIDE_CM_HINTS = {
    k: (v if k != "stride_cm" else 16) for k, v in DEFAULT_MATMUL_HINTS.items()
}
STRIDE_AM_HINTS = {
    k: (v if k != "stride_am" else 16) for k, v in DEFAULT_MATMUL_HINTS.items()
}
STRIDE_CM_AM_HINTS = {
    k: (v if k != "stride_cm" and k != "stride_am" else 16)
    for k, v in DEFAULT_MATMUL_HINTS.items()
}

DEFAULT_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32:1, i32, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_CM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32:16, i32:1, i32, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_AM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32:1, i32:16, i32:1, i32:16, i32:1, 16, 16, 16"
STRIDE_CM_AM_SIGNATURE = "*fp32:16, *fp16:16, *fp16:16, i32, i32, i32, i32:16, i32:1, i32:16, i32:1, i32:16, i32:1, 16, 16, 16"
NO_HINT_SIGNATURE = (
    "*fp32, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 16, 16, 16"
)
