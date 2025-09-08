# QUNET

Single-engine solution for U-Net neural network decoder for Quantum Error Correction (QEC), implemented with quantization-aware training (using Brevitas) and deployed on FPGA using Vitis HLS and the FINN framework. The model supports an early-exit mechanism to reduce average inference latency.

## Repository Structure

### `hw_src/`

Contains the hardware source files used to build the quantized U-Net model on the FPGA using Vitis HLS. The layer functions (MVAU, VVAU, streaming convolution, sliding window, upsampling, etc.) are sourced from [finn-hlslib](https://github.com/Xilinx/finn-hlslib), Xilinx's open-source HLS library for FINN-generated accelerators.

- **`design_src/NNStreaming.cpp`** — Top-level HLS function. Defines the streaming interface (`top`) that accepts input feature maps, runtime weight and threshold streams, and control signals (e.g. `block`, `ee_flag` for early exit). Uses AXI-Stream ports for data and AXI-Lite for control.
- **`design_src/top.hpp`** — Declares the top-level function signature and hardware parameters.
- **`dma_src/`** — DMA source files for streaming input data (`idma`) and reading output results (`odma`) to/from the accelerator.

### `sw_src/`

Contains Python files for quantum circuit generation, dataset sampling, and model definition.

- **`EE_model.py`** — Defines the quantized U-Net architecture (using Brevitas `QuantConv2d`, `QuantReLU`, `BatchNorm2d`) with an early-exit mechanism. Implements progressive model stages (`EA_LAYER_1`, `EA_LAYER_2`, `EA_LAYER_3`) where each stage adds more encoder/decoder depth. Includes an inference-mode wrapper that exits early on easy samples.
- **`detector_error_model.py`** — Generates quantum error correction circuits using `stim` for the rotated planar code. Defines the noise model and circuit structure (data qubits, ancilla qubits, X/Z stabilizer measurements) for sampling syndrome data.
- **`unet_3L_FINN.ipynb`** — Notebook for exporting and converting the trained quantized U-Net model to the FINN ONNX format for hardware synthesis.

### `driver/`

Contains the Python driver needed to run the model on the FPGA.

- **`driver_base.py`** — Base driver class (`FINNExampleOverlay`) built on top of PYNQ's `Overlay`. Handles bitstream loading, DMA buffer allocation, clock configuration, runtime weight loading from `.npy` files, and data packing/unpacking using the FINN data packing utilities.
- **`driver.ipynb`** — Notebook for running end-to-end inference on the FPGA: loading the overlay, preparing syndrome inputs, invoking the accelerator, and evaluating decoding accuracy.
- **`MVAU_weights_npy/`** — Pre-quantized MVAU layer weights stored as `.npy` files, loaded at runtime into the accelerator via the driver.
- **`finn/`**, **`qonnx/`** — Local copies of FINN and QONNX utility modules for data type handling and tensor packing, used by the driver.
