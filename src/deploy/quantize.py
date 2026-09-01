"""
INT8 quantize the exported ONNX model for CPU inference on Raspberry Pi 5.

Owner: hardware track. Run this right after export_onnx.py works.
"""

# from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(onnx_path: str, out_path: str):
    # TODO:
    # quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    raise NotImplementedError


if __name__ == "__main__":
    pass
