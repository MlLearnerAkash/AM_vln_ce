# Building and running VLN-CE with Docker

This document describes how to build and run the VLN-CE repository inside Docker containers.

Notes:
- The project targets Python 3.6 and uses Habitat-Sim (v0.1.7) which is recommended to be installed via conda.
- The Dockerfiles included are best-effort: they attempt to install Habitat-Sim via conda channels and install Python requirements. Some components (Habitat-Sim, habitat-lab) may require additional system dependencies or a custom build for your GPU/host.


Build (CPU, Python 3.9, pip):
```bash
docker build -t vlnce:cpu -f Dockerfile .
```

Build (GPU, Python 3.9, pip):
```bash
docker build -t vlnce:gpu -f Dockerfile.gpu .
# For GPU runtime, use nvidia-container-toolkit (docker run --gpus all ...)
```

Using docker-compose (bind-mounts `./data` into the container):
```bash
docker-compose build
docker-compose run --rm vlnce
```

After starting a shell in the container, the image uses system Python 3.9.

Example: run training/evaluation as in the README
```bash
python run.py --exp-config vlnce_baselines/config/r2r_baselines/nonlearning.yaml --run-type eval
```

Troubleshooting:
- If Habitat-Sim fails to install via conda, follow the instructions in `habitat-sim`'s README to build from source and then re-run the habitat-lab installation step inside the container.
- TensorFlow pinned to 1.13.1 may be incompatible with newer Python versions; the Dockerfiles create a Python 3.6 conda env to match repository expectations.

Symlinked datasets (soft links)
- If your dataset directories are present as symlinks on the host, Docker bind-mounts will preserve symlinks that point inside the mounted path. Example: if `data/datasets` contains a relative symlink `rxr -> ../external/rxr` and you mount the repository root, the link will work inside the container.
- If symlinks are absolute (e.g. `/mnt/data/rxr`) then you must also mount the symlink target inside the container at the same path, for example:

```bash
docker run --rm -it \
	-v /data/ws/VLN-CE:/workspace \
	-v /mnt/data:/mnt/data \
	vlnce:cpu /bin/bash
```

- Alternative: replace symlinks with bind-mounts at runtime or recreate symlinks inside the container after mounting the datasets. Example (inside container):

```bash
mkdir -p /workspace/data/datasets/RxR_VLNCE_v0/text_features
ln -s /mnt/data/rxr_text_features /workspace/data/datasets/RxR_VLNCE_v0/text_features
```

- If you prefer docker-compose, specify the same host mounts under `volumes:` so absolute symlink targets exist inside the container.

Notes on compatibility:
- The repository originally targeted Python 3.6 and uses Habitat-Sim v0.1.7 (conda packaging). Building Habitat-Sim from source or installing its binary distribution may still be required for full functionality (especially GPU builds).
- `tensorflow==1.13.1` from `requirements.txt` may not install cleanly on Python 3.9. If you hit issues, consider using a Python 3.6 image or creating an isolated environment for TensorFlow, or upgrading TensorFlow (may require code changes).
