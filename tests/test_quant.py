import numpy as np
from EAFL_ALP.quant.vq import VQQuant

def test_quant_nonempty():
    vq = VQQuant()
    x  = np.random.randn(4096).astype(np.float32)
    meta = vq.quant(x.copy(), None)
    assert meta, "meta 结果不能为空"
