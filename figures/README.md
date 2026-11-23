# Reconstruction and gradient checks

To visualize a reconstruction using only the bundled STL, run these commands from the repository root. The second command writes `runs/example/comparison.png`.

```sh
uv run python optimize.py
uv run python helpers/plot_sweep.py runs/example
```

## Gradient validation

These scripts require experiment-specific inputs supplied by the caller.

| Script | Inputs | Output |
|---|---|---|
| `make_gradcheck_figure.py` | Scene SDF and probe poses | Single- and two-bounce gradient comparison |
| `gradcheck_convergence.py` | Scene SDF and probe poses | Gradient convergence across sampling rates and resolutions |

For a fresh gradient comparison, supply the scene’s SDF grid in the renderer’s NPY format at `data/vertebrae/vertebrae.npy` and its probe transforms at `data/vertebrae/poses.npy`. The default uses pose index 121 and the archived vertebra probe, with a Gaussian PSF of 2 pixels on each axis. A scene-specific `td_setting.json` overrides the probe settings. Then run

```sh
uv run python figures/make_gradcheck_figure.py --out figures/fresh_gradients
```

With the original experiment inputs and the [paper configuration](../configs/gradient_validation.json), the verified cosine similarities were 0.9670 for one bounce and 0.9523 for two.

## Phantom replay comparison

Across the 603-frame phantom acquisition, the mesh from the [real-data reconstruction](../README.md#optional-real-data-reconstruction) had a symmetric mean sampled surface distance of **0.096 mm** from the archived reconstruction. Both meshes use tracker coordinates.
