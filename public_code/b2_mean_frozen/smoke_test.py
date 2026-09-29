import numpy as np
import torch

from b2_mean import B2Mean
from train_b2_mean import physical_recording_window_weights


def main():
    torch.manual_seed(1)
    model = B2Mean(n_classes=2)
    x = torch.randn(4, 3, 2, 128, 64)
    out = model(x)
    assert out["z"].shape == (4, 3, 64)
    assert out["fused"].shape == (4, 64)
    assert out["logits"].shape == (4, 2)
    assert torch.allclose(out["fused"], out["z"].mean(dim=1))
    assert torch.allclose(out["gate_weights"], torch.full((4, 3), 1.0 / 3.0))

    sample_ids = np.array(["R1", "R1", "R2", "R3", "R3", "R3"])
    labels = np.array(["A", "A", "A", "B", "B", "B"])
    w = physical_recording_window_weights(sample_ids, labels)
    assert np.isfinite(w).all() and abs(float(w.mean()) - 1.0) < 1e-6
    print("B2_MEAN_PUBLIC_SMOKE_PASS")


if __name__ == "__main__":
    main()
