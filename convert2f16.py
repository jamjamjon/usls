import onnx
from pathlib import Path
from onnxconverter_common import float16

model_f32 = "onnx_model.onnx"
model_f16 = float16.convert_float_to_float16(onnx.load(model_f32))
saveout = Path(model_f32).with_name(Path(model_f32).stem + "-f16.onnx")
onnx.save(model_f16, saveout)
