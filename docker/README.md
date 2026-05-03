.env file:
```
# CUDA version (default: 12.1.0)
CUDA_VERSION=12.4.1
NVIDIA_IMAGE_VERSION=12.4.1-devel-ubuntu22.04

# Which GPU to use (default: 0)
CUDA_VISIBLE_DEVICES=0

# OpenAI API key (optional)
OPENAI_API_KEY=your_key_here
```


```
docker-compose up -d --build

# Start everything
docker-compose up -d

# Open development shell
docker-compose exec app bash

# Run Streamlit (from inside container)
streamlit run /workspace/Wikontic/Wikontic.py --server.port=8501 --server.address=0.0.0.0

# View logs
docker-compose logs -f app

# Stop everything
docker-compose down

# Complete reset (deletes all data)
docker-compose down -v
```

echo ""
echo "✅ Development environment ready!"
echo ""
echo "Access services:"
echo "  Streamlit: http://localhost:8501"
echo "  Jupyter:   http://localhost:8888"
echo "  MongoDB:   localhost:27018"
echo ""
echo "Open a shell: docker-compose exec app bash"
echo "View logs:    docker-compose logs -f"
echo "Stop:         docker-compose down"