# Portable ONNX Inference Manual Checklist

This checklist is required before trusting ONNX Runtime output for research or clinical analysis.

## Manual Validation Steps

- [ ] 1. Confirm standard PyTorch nnU-Net prediction still works.
  - Command:
    ```bash
    nnUNetv2_predict -i imagesTs -o output_pytorch -d DatasetXXX -c 3d_fullres -f all
    ```
  - Notes:

- [ ] 2. Export ONNX from `fold_all/checkpoint_final.pth`.
  - Command:
    ```bash
    nnUNetv2_export_onnx \
      --model_dir /path/to/model_dir \
      --output_onnx /path/to/export/model.onnx \
      --checkpoint checkpoint_final.pth
    ```
  - Notes:

- [ ] 3. Confirm `model.onnx` and `model_export.json` are created.
  - `model.onnx` path:
  - `model_export.json` path:

- [ ] 4. Run ONNX checker.
  - Confirm the export command completed its ONNX checker step.
  - Notes:

- [ ] 5. Validate random patch equivalence.
  - Command:
    ```bash
    nnUNetv2_validate_onnx_random_patch \
      --model_dir /path/to/model_dir \
      --onnx_model /path/to/export/model.onnx
    ```
  - Result:

- [ ] 6. Validate real preprocessed patch equivalence.
  - Command:
    ```bash
    nnUNetv2_validate_onnx_preprocessed_patch \
      --model_dir /path/to/model_dir \
      --onnx_model /path/to/export/model.onnx \
      --preprocessed_patch /path/to/preprocessed_patch.npy
    ```
  - Result:

- [ ] 7. Run PyTorch prediction on one case.
  - Command:
    ```bash
    nnUNetv2_predict -i one_case_imagesTs -o one_case_pytorch -d DatasetXXX -c 3d_fullres -f all
    ```
  - Output segmentation:

- [ ] 8. Run ONNX Runtime prediction on the same case.
  - Command:
    ```bash
    nnUNetv2_predict \
      -i one_case_imagesTs \
      -o one_case_onnxruntime \
      -d DatasetXXX \
      -c 3d_fullres \
      -f all \
      --backend onnxruntime \
      --onnx_model /path/to/export/model.onnx \
      --ort_provider CPUExecutionProvider
    ```
  - Output segmentation:

- [ ] 9. Compare final segmentations.
  - Command:
    ```bash
    nnUNetv2_compare_segmentations \
      --seg_a /path/to/one_case_pytorch/case.nii.gz \
      --seg_b /path/to/one_case_onnxruntime/case.nii.gz \
      --output_json one_case_comparison.json
    ```
  - Result:

- [ ] 10. Confirm Dice per label between PyTorch and ONNX outputs is acceptable.
  - Acceptance criteria:
  - Observed Dice per label:

- [ ] 11. Confirm voxel disagreement is acceptable.
  - Acceptance criteria:
  - Observed voxel disagreement:

- [ ] 12. Benchmark runtime.
  - Command:
    ```bash
    python scripts/benchmark_nnunet_speed_knobs.py \
      --input imagesTs \
      --output speed_knob_benchmark \
      --model_dir /path/to/model_dir
    ```
  - PyTorch runtime:
  - ONNX Runtime runtime, if measured separately:

- [ ] 13. Export ONNX bundle.
  - Command:
    ```bash
    nnUNetv2_export_onnx_bundle \
      --model_dir /path/to/model_dir \
      --onnx_model /path/to/export/model.onnx \
      --output_bundle exported_bundle
    ```
  - Bundle path:

- [ ] 14. Run bundle script on `CPUExecutionProvider`.
  - Command:
    ```bash
    python run_nnunet_onnx.py \
      --bundle exported_bundle \
      --input imagesTs \
      --output bundle_output \
      --provider CPUExecutionProvider
    ```
  - Result:

- [ ] 15. Confirm `prediction_manifest.json` is saved.
  - Manifest path:
  - Notes:

- [ ] 16. Confirm default `nnUNetv2_predict` behavior is unchanged.
  - Re-run the default command without `--backend onnxruntime`.
  - Confirm output matches the expected PyTorch baseline.
  - Notes:

- [ ] 17. Confirm training commands are unchanged.
  - Confirm no training command behavior was modified for this validation.
  - Notes:

- [ ] 18. Record package versions.
  - Python:
  - torch:
  - nnunetv2:
  - onnx:
  - onnxruntime:
  - numpy:

- [ ] 19. Record hardware.
  - CPU:
  - GPU, if used:
  - RAM:

- [ ] 20. Document any observed differences.
  - Numerical differences:
  - Segmentation differences:
  - Runtime differences:
  - Visual review findings:
  - Follow-up actions:
