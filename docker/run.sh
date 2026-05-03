#!/bin/bash

# Default CUDA version if not specified
CUDA_VERSION=${1:-12.1.0}
IMAGE_NAME="ppr-wikontic-hybrid"
IMAGE_TAG="cuda-${CUDA_VERSION}"
CONTAINER_NAME="ppr-wikontic-${CUDA_VERSION//./-}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting container with CUDA ${CUDA_VERSION}..."
echo "Mounting project root: ${PROJECT_ROOT} -> /workspace/PPR_for_Wikontic"

# Check if nvidia-docker is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
    GPU_FLAGS=""
else
    GPU_FLAGS="--gpus all"
    echo "GPU support enabled"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs echo "   GPU:"
fi

# Check if container already exists and remove it
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f ${CONTAINER_NAME}
fi

# Run the container
docker run ${GPU_FLAGS} \
    -it \
    --rm \
    --name ${CONTAINER_NAME} \
    -v ${PROJECT_ROOT}:/workspace/PPR_for_Wikontic \
    -v ${PROJECT_ROOT}/data:/data \
    -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    -e PYTHONPATH="/workspace:/workspace/HippoRAG:/workspace/Wikontic:/workspace/PPR_for_Wikontic" \
    -p 8501:8501 \
    --shm-size=8gb \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    ${@:2}  # Pass any additional commands

# Note: ${@:2} allows you to run commands like:
# ./run.sh 12.1.0 "python HippoRAG/demo.py"
# ./run.sh 12.1.0 "streamlit run Wikontic/app.py --server.port=8501 --server.address=0.0.0.0"