import torch

from tlsseg.runtime import configure_precision, environment


def test_precision_policy_disables_both_tf32_paths():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configure_precision()
    assert torch.get_default_dtype() == torch.float32
    assert torch.get_float32_matmul_precision() == "highest"
    precision = environment()["precision"]
    assert precision["cuda_matmul_allow_tf32"] is False
    assert precision["cudnn_allow_tf32"] is False
