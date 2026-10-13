"""Print an ONNX model's tensor contract and embedded metadata without inference."""

import argparse
import json
from pathlib import Path

import onnx


def tensor_info(value):
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        ]
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to an ONNX model")
    args = parser.parse_args()
    model = onnx.load(args.model, load_external_data=False)
    report = {
        "model": str(args.model.resolve()),
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": [
            {"domain": item.domain or "ai.onnx", "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [tensor_info(value) for value in model.graph.input],
        "outputs": [tensor_info(value) for value in model.graph.output],
        "metadata": {item.key: item.value for item in model.metadata_props},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
