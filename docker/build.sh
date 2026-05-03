#!/bin/bash

# Default CUDA version if not specified
CUDA_VERSION=${1:-12.1.0}
IMAGE_NAME="ppr-wikontic-hybrid"
IMAGE_TAG="cuda-${CUDA_VERSION}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building with CUDA ${CUDA_VERSION}..."
echo "Project root: ${PROJECT_ROOT}"
echo "Script directory: ${SCRIPT_DIR}"

# Validate CUDA version
case $CUDA_VERSION in
    11.8.0|12.1.0|12.2.0|12.3.0|12.4.0|12.5.0|13.1.0)
        echo "Using CUDA ${CUDA_VERSION}"
        ;;
    *)
        echo "Warning: CUDA ${CUDA_VERSION} may not be supported. Available: 11.8.0, 12.1.0, 12.2.0, 12.3.0, 12.4.0, 12.5.0, 13.1.0"
        ;;
esac

# Build the Docker image
docker build \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg PYTHON_VERSION=3.10 \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f ${SCRIPT_DIR}/Dockerfile \
    ${SCRIPT_DIR}

echo "Built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run: ./run.sh ${CUDA_VERSION}"