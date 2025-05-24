# ray_examples
Various ray example projects

## 📁 Project Overview

This repository contains comprehensive examples demonstrating different aspects of Ray, the distributed computing framework. Each folder focuses on specific Ray capabilities and use cases.

## 🚀 Projects

### 📊 `distributed-k-means/`
**Advanced GPU-Accelerated Distributed Machine Learning**

A comprehensive project showcasing FAISS GPU-accelerated K-means clustering integrated with Ray cluster management and PyTorch DDP training.

**Key Features:**
- GPU-accelerated clustering using FAISS with 10-100x speedup over CPU
- Distributed processing across Ray workers with fault tolerance
- Hybrid ML workflows combining PyTorch DDP training with FAISS clustering
- Production-ready architecture with automatic GPU memory management
- Supports millions of high-dimensional vectors with multi-GPU scaling

**Files:**
- `faiss_distributed_kmeans_gpu.py` - Pure FAISS GPU k-means clustering
- `pytorch_ddp_faiss_hybrid.py` - Hybrid PyTorch DDP + FAISS approach
- `README.md` - Detailed documentation with benchmarks and architecture diagrams

### 🎭 `actors/`
**Ray Actors - Stateful Distributed Computing**

Examples demonstrating Ray's actor model for stateful distributed computation.

**Examples:**
- `ray_core_actors.py` - Basic counter actor demonstrating state management
- `map_example.py` - Actor pool pattern for distributed task processing
- `actor_async_method.py` - Async actors with concurrent task handling and thread pools

**Use Cases:** Stateful services, distributed caching, parallel processing with shared state

### ⚡ `functions/`
**Ray Functions - Stateless Distributed Computing**

Examples of Ray's core remote function capabilities for stateless parallel processing.

**Examples:**
- `ray_core_function.py` - Basic remote functions and parallel execution
- `sleepy_task.py` - Performance comparison: sequential vs. parallel execution
- `web_crawler.py` - Recursive web crawling using distributed functions

**Use Cases:** Embarrassingly parallel computations, data processing pipelines, web scraping

### 🌐 `serve/`
**Ray Serve - Model Serving and Web Services**

Demonstrates Ray Serve for deploying machine learning models and web services.

**Examples:**
- `deployment_example.py` - Temperature converter service with multiple replicas and resource allocation

**Use Cases:** ML model serving, API endpoints, microservices architecture

### 🔧 `utility/`
**Ray Cluster Management Utilities**

Helper scripts for managing and monitoring Ray clusters.

**Tools:**
- `show_ray_nodes.py` - Display cluster node information, resources, and health status

**Use Cases:** Cluster monitoring, debugging, resource management

## 🛠️ Getting Started

1. **Install Ray:**
   ```bash
   pip install ray[default]
   ```

2. **For GPU examples (distributed-k-means):**
   ```bash
   pip install -r distributed-k-means/requirements.txt
   ```

3. **Run any example:**
   ```bash
   python <folder>/<example_file>.py
   ```

## 💡 Learning Path

1. **Start with:** `functions/ray_core_function.py` - Learn basic Ray concepts
2. **Then try:** `actors/ray_core_actors.py` - Understand stateful computation
3. **Scale up:** `actors/map_example.py` - Actor pools for distributed processing
4. **Web services:** `serve/deployment_example.py` - Deploy services with Ray Serve
5. **Advanced:** `distributed-k-means/` - Production-ready GPU-accelerated ML workflows

## 🎯 Use Case Matrix

| Project | Stateless | Stateful | GPU | Web Serving | Production Ready |
|---------|-----------|----------|-----|-------------|------------------|
| functions/ | ✅ | ❌ | ❌ | ❌ | 🟡 |
| actors/ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| serve/ | ✅ | ✅ | ❌ | ✅ | ✅ |
| distributed-k-means/ | ❌ | ✅ | ✅ | ❌ | ✅ |
| utility/ | ✅ | ❌ | ❌ | ❌ | ✅ |
