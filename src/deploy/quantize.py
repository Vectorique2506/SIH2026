"""
INT8 quantize the exported ONNX model for fast CPU inference
(laptop AI stage, and any future Pi-side AI attempt).

Run from project root:
    python src/deploy/quantize.py
"""

import os

from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "models_exported/sih_anc_model.onnx"
OUTPUT_MODEL = "models_exported/sih_anc_model_int8.onnx"


def _total_model_size(onnx_path: str) -> int:
    """Sum the .onnx graph file plus any external .onnx.data weight file,
    since PyTorch/ONNX exports sometimes split weights into a separate file."""
    total = os.path.getsize(onnx_path)
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total


def quantize_model(onnx_path: str, out_path: str):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"{onnx_path} not found. Make sure the exported ONNX model "
            f"(and its .data file, if any) are in models_exported/."
        )

    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = _total_model_size(onnx_path) / 1024
    new_size = _total_model_size(out_path) / 1024
    print(f"Original model (graph + weights): {orig_size:.1f} KB")
    print(f"Quantized model: {new_size:.1f} KB")
    print(f"Size reduction: {100 * (1 - new_size / orig_size):.1f}%")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    quantize_model(INPUT_MODEL, OUTPUT_MODEL)
    
    
    