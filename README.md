# ComfyUI-TrellisMeshPostprocess

ComfyUI custom node for Trellis mesh normal postprocessing.

## Node

- `Trellis2 - Mesh Postprocess Normals`
- Class: `TamataTrellisMeshPostprocessNormals`
- Input: `TRIMESH`
- Output: `TRIMESH`

This node recalculates normals by clustering face normals in quantized position space. It is designed for Trellis outputs where duplicated/split vertices can produce hard lighting seams.

## Inputs

- `trimesh`: mesh from Trellis texturing/refinement stages.
- `position_epsilon` (default `1e-5`): weld tolerance used only for normal clustering.
- `normal_crease_deg` (default `55`): smoothing angle threshold.

## Install

Clone into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/Dodzilla/ComfyUI-TrellisMeshPostprocess.git
```

Restart ComfyUI.

## License

MIT
