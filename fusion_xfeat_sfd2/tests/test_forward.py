import torch

def test_dummy_forward():
    # 占位测试：未来替换为实际模型前向
    x = torch.randn(1, 3, 224, 224)
    assert x.shape[0] == 1
