# Structure

- `search-server/`: Contains the code for the search server, which handles image and text queries to retrieve relevant map data.
- `segment3d/`: Contains the code for segmenting 3D point clouds into meaningful components and captioning them.
- `semantic-3d-search-demo/`: User interface for demonstrating the search capabilities of the system.

# Search Server

## Docker-based installation

1. Install Docker Engine. For Ubuntu, follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
2. Install the NVIDIA Container Toolkit. For Ubuntu, follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt).
3. Find your host CUDA version:
   ```bash
   nvidia-smi  # check "CUDA Version" in the top-right corner
   ```
4. Set your `OPENAI_API_KEY` in `search-server/.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Running the server

```bash
docker compose up --detach
```

By default, the image is built for **CUDA 13.0** with PyTorch `cu130` wheels. To override for a different driver:

```bash
CUDA_VERSION=12.8.1 TORCH_CUDA_VERSION=cu128 docker compose up --build --detach
```

`CUDA_VERSION` must match the tag of the `nvidia/cuda` base image (e.g. `12.8.1`, `13.0.1`).  
`TORCH_CUDA_VERSION` must be the corresponding PyTorch wheel suffix (e.g. `cu128`, `cu130`).

To pre-load CLIP at startup for a specific dataset (enables CLIP ViT-H-14 search):

```bash
DATASET_NAME=<dataset> docker compose up --detach
```

To print logs: `docker compose logs -f`  
To shut down: `docker compose down`

**Note**: If you're making code changes, rebuild and recreate the container with:

```bash
docker compose up --detach --build --force-recreate --renew-anon-volumes
```
