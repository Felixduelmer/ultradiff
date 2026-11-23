# UltraDiff: Differentiable Ray Tracing in Ultrasound for Shape Optimization

*SIGGRAPH Asia 2026 Technical Communications* · [Paper](https://doi.org/10.1145/3829339.3847840) · [arXiv](https://arxiv.org/abs/2610.07941) · [Citation](#citation)

[Felix Duelmer](https://www.cs.cit.tum.de/camp/members/felix-duelmer/),
[Magdalena Wysocki](https://www.cs.cit.tum.de/camp/members/magdalena-wysocki/),
[Nassir Navab](https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/),
[Mohammad Farid Azampour](https://www.cs.cit.tum.de/camp/members/mohammad-farid-azampour/)

Technical University of Munich & Munich Center for Machine Learning

UltraDiff reconstructs a 3D surface from posed B-mode ultrasound images by fitting a signed distance field to the observed echoes.

## Install

On a machine with a CUDA-capable NVIDIA GPU, install uv and run the following from the repository root. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync
```

## Run the bundled example

The example generates 18 synthetic views of the bundled vertebra STL and writes the fitted surface to `runs/example/reconstruction.stl` in the input mesh’s coordinate frame.

```sh
uv run python optimize.py
```

Use `--mesh path/to/model.stl --out runs/my-mesh` for another STL. Choose a fresh output directory for each run. [Default settings](configs/ultrasound.json) · [Reconstruction and gradient checks](figures/README.md)

## Optional real-data reconstruction

Supply the real phantom acquisition separately in a directory containing `images_train.npy`, `poses_train.npy`, and `td_setting.json`. The phantom extra builds fastsweep and requires a C++ compiler, CMake, and a CUDA toolkit.

```sh
uv run --extra phantom python optimize.py --acquisition /path/to/spine_phantom \
  --config configs/phantom.json --out runs/phantom-replay
```

Replay reads probe geometry from the recording and PSF widths from the preset. The preset fits all 603 native-resolution frames and exports the mesh in tracker coordinates (mm). [Phantom settings](configs/phantom.json)

## Citation

```bibtex
@inproceedings{duelmer2026ultradiff,
  author    = {Duelmer, Felix and Wysocki, Magdalena and Navab, Nassir and Azampour, Mohammad Farid},
  title     = {{UltraDiff}: Differentiable Ray Tracing in Ultrasound for Shape Optimization},
  booktitle = {SIGGRAPH Asia 2026 Technical Communications},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3829339.3847840}
}
```

## Credits

The example mesh is derived from VerSe case 048, label 23. Credit Sekuboyina et al. and the VerSe collaborators when using it. [Dataset](https://github.com/anjany/verse) · [Benchmark paper](https://doi.org/10.1016/j.media.2021.102166) · [Mesh license (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

The renderer builds on the relaxed-boundary SDF renderer by Zichen Wang and collaborators. [Upstream project](https://zichenwang01.github.io/relaxed_boundary/) · [Code license](LICENSE)
