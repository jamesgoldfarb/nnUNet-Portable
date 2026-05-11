# Portable ONNX Inference

## 1. Overview

This workflow keeps nnU-Net as the training, planning, preprocessing, sliding-window inference, resampling, postprocessing, and export engine. ONNX Runtime is used only for the network forward pass on each fixed-size patch.

The ONNX model does not replace nnU-Net. It does not load raw images, normalize data, tile images, blend overlapping windows, resample predictions, apply postprocessing, or write final segmentations. Those steps remain nnU-Net responsibilities.

Users must validate the ONNX Runtime backend for their own model, data, hardware, and provider before using the outputs for research or clinical analysis. This workflow is not a clinical-readiness claim.

## 2. Supported MVP Scope

- nnUNetv2 only
- `3d_fullres` only
- `fold_all/checkpoint_final.pth`
- Fixed patch size only
- Raw logits ONNX export only

## 3. What Is Not Supported Yet

- Standalone no-nnU-Net inference
- Dynamic-shape ONNX
- Multi-fold ensembles
- Cascaded models
- Quantization
- TensorRT engine generation
- Docker deployment

## 4. Exporting ONNX

Export the trained `fold_all` network forward pass to a fixed-shape ONNX model:

```bash
nnUNetv2_export_onnx \
  --model_dir /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --output_onnx /path/to/export/model.onnx \
  --checkpoint checkpoint_final.pth \
  --opset 17 \
  --device cpu
```

The export writes raw logits only. It also writes `model_export.json` next to `model.onnx` when the export succeeds.

## 5. Validating Random Patch Equivalence

Compare PyTorch and ONNX Runtime on the same deterministic random input patch:

```bash
nnUNetv2_validate_onnx_random_patch \
  --model_dir /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --onnx_model /path/to/export/model.onnx \
  --checkpoint checkpoint_final.pth \
  --provider CPUExecutionProvider \
  --seed 12345
```

This checks patch-level logits before any nnU-Net sliding-window or postprocessing steps.

## 6. Validating Real Preprocessed Patch Equivalence

Compare PyTorch and ONNX Runtime on a real preprocessed patch saved as `.npy` or as `.npz` with a `data` array:

```bash
nnUNetv2_validate_onnx_preprocessed_patch \
  --model_dir /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --onnx_model /path/to/export/model.onnx \
  --preprocessed_patch /path/to/preprocessed_patch.npy \
  --checkpoint checkpoint_final.pth \
  --provider CPUExecutionProvider
```

The patch shape must match the exported model shape: `[1, C, D, H, W]` or `[C, D, H, W]` before the batch dimension is added.

## 7. Running Prediction With ONNX Runtime Backend

Run the standard nnU-Net prediction pipeline while replacing only the patch-level network forward call with ONNX Runtime:

```bash
nnUNetv2_predict \
  -i imagesTs \
  -o output_onnx \
  -d DatasetXXX \
  -c 3d_fullres \
  -f all \
  --backend onnxruntime \
  --onnx_model /path/to/export/model.onnx \
  --ort_provider CPUExecutionProvider
```

Omitting `--backend onnxruntime` keeps the default PyTorch backend.

## 8. Comparing PyTorch vs ONNX Runtime Outputs

Run both backends on the same input and compare final segmentations:

```bash
nnUNetv2_compare_backends \
  --input imagesTs \
  --output backend_comparison \
  --model_dir /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --configuration 3d_fullres \
  --fold all \
  --checkpoint checkpoint_final.pth \
  --onnx_model /path/to/export/model.onnx \
  --ort_provider CPUExecutionProvider
```

This writes PyTorch outputs to `backend_comparison/pytorch`, ONNX Runtime outputs to `backend_comparison/onnxruntime`, and a final `comparison_report.json` to `backend_comparison`.

## 9. Exporting a Bundle

Create a minimal portable bundle around an already-exported ONNX model:

```bash
nnUNetv2_export_onnx_bundle \
  --model_dir /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --onnx_model /path/to/export/model.onnx \
  --output_bundle exported_bundle \
  --checkpoint checkpoint_final.pth \
  --configuration 3d_fullres \
  --fold all
```

The bundle contains `model.onnx`, optional `model_export.json`, `plans.json`, `dataset.json`, `inference_config.json`, and `README.md`. It does not include PyTorch weights, training data, Docker assets, quantization artifacts, or standalone preprocessing.

## 10. Running From a Bundle

The bundle still requires an installed `nnunetv2` environment. If your deployment includes a helper script named `run_nnunet_onnx.py`, a typical invocation is:

```bash
python run_nnunet_onnx.py \
  --bundle exported_bundle \
  --input imagesTs \
  --output output_bundle_run \
  --provider CPUExecutionProvider
```

The helper should use the bundle metadata and call nnU-Net for image loading, preprocessing, sliding-window inference support, resampling, postprocessing, and export. It should use ONNX Runtime only for patch-level network forward inference.

## 11. Recommended Validation Before Clinical/Research Use

Before trusting ONNX Runtime output for a model and dataset, run and record:

- Random patch comparison
- Real preprocessed patch comparison
- Full-case segmentation comparison
- Dice per label
- Voxel disagreement
- Visual review

Acceptable thresholds depend on the model, labels, image domain, hardware, provider, and intended use. Users must define and document acceptance criteria for their own work.

## 12. Troubleshooting

### Provider Unavailable

If ONNX Runtime reports that a provider is unavailable, list providers with the validation tools or use `CPUExecutionProvider`. Make sure the installed `onnxruntime` package supports the requested provider.

### Shape Mismatch

The MVP export uses fixed patch size and no dynamic axes. Confirm that the ONNX model was exported from the same nnU-Net configuration, fold, checkpoint, patch size, and input channel count that prediction uses.

The ONNX model input shape must exactly match the patch tensor shape that `nnUNetv2_predict` sends to the network. Do not rely on spatial-axis transposes to compensate for a mismatch; anisotropic kernels or strides can make that a different computation. Re-export the ONNX model with the exact patch shape used by nnU-Net prediction.

### ONNX Export Failure

Check the printed network class, input shape, opset, package versions, and full exception. Re-run export with the default CPU device first, then compare against provider-specific behavior later.

### Output Differs From PyTorch

Run validation in this order: random patch, real preprocessed patch, full-case comparison, then visual review. Differences can come from provider kernels, dtype conversions, export limitations, mismatched model metadata, or an ONNX model exported from different weights.

### Missing `nnunetv2` Dependency

The ONNX model is not a standalone inference application. Install a compatible `nnunetv2` environment so nnU-Net can still perform preprocessing, sliding-window inference support, resampling, postprocessing, and export.
