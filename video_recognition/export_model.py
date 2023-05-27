import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

RUN_NAME = "levit_conv_128"
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / f"../runs/{RUN_NAME}/best_model.torchscript"
DEFAULT_ONNX_PATH = Path(__file__).parent / f"../runs/{RUN_NAME}/model.onnx"
DEFAULT_OPENVINO_PATH = Path(__file__).parent / f"../runs/{RUN_NAME}/openvino_model"


def export_onnx(weights_path=DEFAULT_WEIGHTS_PATH, onnx_path=DEFAULT_ONNX_PATH):
    model = torch.jit.load(weights_path, map_location="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    # torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # check conversation results
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # compute pytorch model outputs
    with torch.no_grad():
        torch_model_outs = model(dummy_input).numpy()

    np.testing.assert_allclose(
        torch_model_outs,
        ort_outs[0],
        rtol=1e-03,
        atol=1e-05,
    )


def export_openvino(
    weights_path=DEFAULT_ONNX_PATH, openvino_path=DEFAULT_OPENVINO_PATH
):
    mo_command = f"""mo
                    --input_model "{weights_path}"
                    --compress_to_fp16
                    --output_dir "{openvino_path}"
                    """
    mo_command = " ".join(mo_command.split())
    os.system(mo_command)


if __name__ == "__main__":
    export_onnx()
    export_openvino()
