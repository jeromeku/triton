import os
import logging 
os.environ["TORCH_LOGS_FORMAT"] = "%(levelname)s %(pathname)s:%(lineno)d %(message)s"
import torch

from torch._logging import set_logs
from torch._inductor import config as inductor_config

set_logs(inductor=logging.DEBUG, recompiles_verbose=True)
# fmt = logging.Formatter("%(levelname)s %(pathname)s:%(lineno)d %(message)s")
# for h in logging.getLogger("torch").handlers: h.setFormatter(fmt)

@torch.compile
def fn(x):
    return x + 1
torch._dynamo.reset()

with inductor_config.patch(
    max_autotune=True,
    max_autotune_gemm_backends="TRITON",
    autotune_num_choices_displayed=None,
):
    _ = fn(torch.ones(3, 3, device="cuda"))