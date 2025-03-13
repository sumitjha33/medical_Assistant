#!/bin/bash

# Update pip and install dependencies
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Create cache directory
export TRANSFORMER_CACHE="/opt/render/project/src/.cache/huggingface"
mkdir -p $TRANSFORMER_CACHE

# Set environment variables
export SENTENCE_TRANSFORMERS_HOME=$TRANSFORMER_CACHE
export TRANSFORMERS_CACHE=$TRANSFORMER_CACHE
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export TRANSFORMERS_OFFLINE=1

# Clean any existing cache
rm -rf $TRANSFORMER_CACHE/*

# Pre-download only the smaller model
python -c "
from sentence_transformers import SentenceTransformer
import gc
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', cache_folder='$TRANSFORMER_CACHE')
del model
gc.collect()
"
