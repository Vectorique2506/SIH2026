"""
Export trained PyTorch checkpoint to ONNX.

Owner: whoever owns the Pi 5 hardware bring-up — start this EARLY,
not in the last hours. Export failures are the most common reason
a good model never makes it to the demo.
"""

import torch


def export_to_onnx(model, checkpoint_path: str, out_path: str, dummy_shape=(1, 1, 16000)):
    # TODO:
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()
    # dummy_input = torch.randn(*dummy_shape)
    # torch.onnx.export(model, dummy_input, out_path,
    #                    input_names=["noisy"], output_names=["enhanced"],
    #                    dynamic_axes={"noisy": {0: "batch"}, "enhanced": {0: "batch"}})
    raise NotImplementedError


if __name__ == "__main__":
    pass
