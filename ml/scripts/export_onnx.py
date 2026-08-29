"""Export a ControlPlane model artifact to ONNX format for faster CPU inference.

Usage:
  python -m ml.scripts.export_onnx --artifact ml/artifacts/injection-v1

This uses `optimum` to export the PyTorch weights to an ONNX graph, saving it
inside the artifact directory at `<artifact>/model_onnx/`.
It also updates `<artifact>/calibration.json` with `"onnx": true`.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export a model artifact to ONNX")
    parser.add_argument("--artifact", required=True, type=str, help="Path to the artifact directory")
    parser.add_argument("--quantize", action="store_true", help="Also generate an INT8 quantized ONNX model")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact)
    model_dir = artifact_dir / "model"
    onnx_dir = artifact_dir / "model_onnx"
    quantized_dir = artifact_dir / "model_quantized_onnx"
    calib_path = artifact_dir / "calibration.json"

    if not model_dir.exists():
        print(f"[ERROR] No model/ directory found in {artifact_dir}")
        return
    if not calib_path.exists():
        print(f"[ERROR] No calibration.json found in {artifact_dir}")
        return

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] optimum[onnxruntime] is not installed. Run: pip install optimum[onnxruntime]")
        return

    print(f"[INFO] Loading PyTorch model from {model_dir} for ONNX export...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    # Handle PEFT models by merging them first
    if (model_dir / "adapter_config.json").exists():
        print("[INFO] PEFT adapter detected. Merging with base model...")
        import tempfile
        from peft import AutoPeftModelForSequenceClassification
        
        peft_model = AutoPeftModelForSequenceClassification.from_pretrained(str(model_dir))
        merged_model = peft_model.merge_and_unload()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)
            model = ORTModelForSequenceClassification.from_pretrained(tmpdir, export=True)
    else:
        # optimum handles the conversion from PyTorch implicitly when export=True
        model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)

    print(f"[INFO] Saving ONNX model to {onnx_dir}...")
    model.save_pretrained(str(onnx_dir))
    tokenizer.save_pretrained(str(onnx_dir))

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    calib["onnx"] = True

    if args.quantize:
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            print(f"[INFO] Quantizing ONNX model to INT8 at {quantized_dir}...")
            quantizer = ORTQuantizer.from_pretrained(model)
            dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=str(quantized_dir), quantization_config=dqconfig)
            tokenizer.save_pretrained(str(quantized_dir))
            calib["quantized_onnx"] = True
            print("[INFO] INT8 Quantization successful.")
        except Exception as e:
            print(f"[ERROR] Failed to quantize model: {e}")

    print("[INFO] Updating calibration.json...")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    print("[OK] ONNX export complete.")

if __name__ == "__main__":
    main()
