OOP 8-bit (TIDL) outputs on forvia4testset, data-augmentation model (compare-inf 7db6a76)

Same format as your <video>_out_positions.csv: one row per video frame, the 8 class probabilities.

board/         full C++ pipeline on the TI AM62A board (NPU), video decoded on the board
               (OpenCV 4.13 / FFmpeg 8.0) -- this is what the demo image produces
pc_emulation/  same C++ pipeline in TIDL 10_01 PC emulation (docker forvia_oms_library:10_01,
               OpenCV 4.5.4 / FFmpeg 4.4)

Model and artifacts (oop-model branch RD-5790, commit 24c9a0c, MR !5):
  onnx_models/model_positions_no_keypts.onnx                                   MD5 c05760eeac21f0006cbcd1702cbe449b
  model-artifacts/model_positions_without_keypts_device/artifacts/deploy_lib.so  MD5 d778dccdec929ffe3f2a3cfa86f3ac67  (board)
  model-artifacts/model_positions_without_keypts/artifacts/deploy_lib.so         MD5 a1c58749488f6751a37c7e34ca3c0058  (PC emulation)
  Compiled with TIDL 10_01_00_02, calibration shuffle seeded (random.seed(0)): with this seed the board
  NPU gives the same model outputs as the PC emulation.

Against the fp32 Python reference (your inference script, 13076 frames x 8 classes = 104608 decisions):
  board        99.52% same decisions, mean F1 0.7302
  pc_emulation 99.51% same decisions, mean F1 0.7325
  fp32         mean F1 0.7296
  board vs pc_emulation 99.80% (the difference comes from the video decoder, not the model)
Details: Confluence pages 4904878082 (model only) and 4907859970 (full pipeline).
