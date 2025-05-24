#!/usr/bin/env python3
"""
FAISS Distributed K-means with GPU Support in Ray Cluster
=========================================================

This example demonstrates how to use FAISS's built-in distributed k-means
with GPU acceleration in a Ray cluster environment.

Features:
- GPU-accelerated k-means clustering using FAISS
- Distributed computation across multiple Ray workers
- Support for large-scale datasets
- Automatic load balancing and fault tolerance via Ray
"""

import os
import time
import numpy as np
import ray
import faiss
import torch
from typing import List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class FAISSGPUKMeansWorker:
    """
    Ray actor that performs GPU-accelerated k-means clustering using FAISS.
    Each worker handles a portion of the data and uses GPU resources.
    """
    
    def __init__(self, worker_id: int, gpu_id: int = 0):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.gpu_resource = None
        self.index = None
        self.centroids = None
        
        # Initialize GPU resources
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Initialize FAISS GPU resources"""
        try:
            # Create GPU resources
            self.gpu_resource = faiss.StandardGpuResources()
            
            # Configure GPU settings
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = False  # Use float32 for better precision
            config.device = self.gpu_id
            
            logger.info(f"Worker {self.worker_id}: GPU {self.gpu_id} initialized")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to initialize GPU: {e}")
            # Fallback to CPU if GPU initialization fails
            self.gpu_resource = None
    
    def assign_to_centroids(self, data: np.ndarray, centroids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign data points to nearest centroids using GPU-accelerated search.
        
        Args:
            data: Data points to assign (n_points, dim)
            centroids: Current centroids (k, dim) 
            k: Number of clusters
            
        Returns:
            Tuple of (distances, assignments)
        """
        try:
            d = data.shape[1]
            
            if self.gpu_resource is not None:
                # Use GPU index
                config = faiss.GpuIndexFlatConfig()
                config.useFloat16 = False
                config.device = self.gpu_id
                
                # Create GPU index for L2 distance
                index = faiss.GpuIndexFlatL2(self.gpu_resource, d, config)
            else:
                # Fallback to CPU index
                index = faiss.IndexFlatL2(d)
            
            # Add centroids to index
            index.add(centroids.astype(np.float32))
            
            # Search for nearest centroids
            distances, assignments = index.search(data.astype(np.float32), 1)
            
            # Flatten assignments (remove extra dimension)
            assignments = assignments.flatten()
            distances = distances.flatten()
            
            logger.info(f"Worker {self.worker_id}: Assigned {len(data)} points to centroids")
            
            return distances, assignments
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Assignment failed: {e}")
            raise
    
    def compute_local_statistics(self, data: np.ndarray, assignments: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local centroid statistics for this worker's data.
        
        Args:
            data: Local data points
            assignments: Cluster assignments for each point
            k: Number of clusters
            
        Returns:
            Tuple of (centroid_sums, cluster_counts)
        """
        d = data.shape[1]
        centroid_sums = np.zeros((k, d), dtype=np.float64)
        cluster_counts = np.zeros(k, dtype=np.int64)
        
        # Accumulate sums and counts for each cluster
        for i in range(k):
            mask = assignments == i
            if np.any(mask):
                centroid_sums[i] = np.sum(data[mask], axis=0)
                cluster_counts[i] = np.sum(mask)
        
        return centroid_sums, cluster_counts


@ray.remote
class FAISSDistributedKMeansCoordinator:
    """
    Coordinator for distributed k-means clustering using FAISS GPU workers.
    Handles the master logic for iterative k-means updates.
    """
    
    def __init__(self, k: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None
        
        np.random.seed(seed)
    
    def initialize_centroids(self, sample_data: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using k-means++ algorithm.
        
        Args:
            sample_data: Sample of the data for centroid initialization
            
        Returns:
            Initial centroids
        """
        n_samples, d = sample_data.shape
        centroids = np.zeros((self.k, d))
        
        # Choose first centroid randomly
        centroids[0] = sample_data[np.random.randint(n_samples)]
        
        # Choose remaining centroids using k-means++
        for c_id in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:c_id]]) 
                                for x in sample_data])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    i = j
                    break
            
            centroids[c_id] = sample_data[i]
        
        self.centroids = centroids
        logger.info(f"Initialized {self.k} centroids using k-means++")
        return centroids
    
    def update_centroids(self, all_centroid_sums: List[np.ndarray], 
                        all_cluster_counts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Update centroids based on statistics from all workers.
        
        Args:
            all_centroid_sums: List of centroid sums from each worker
            all_cluster_counts: List of cluster counts from each worker
            
        Returns:
            Tuple of (new_centroids, centroid_shift)
        """
        # Aggregate statistics from all workers
        total_sums = np.sum(all_centroid_sums, axis=0)
        total_counts = np.sum(all_cluster_counts, axis=0)
        
        # Compute new centroids
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            if total_counts[i] > 0:
                new_centroids[i] = total_sums[i] / total_counts[i]
            else:
                # Keep old centroid if no points assigned
                new_centroids[i] = self.centroids[i]
        
        # Compute centroid shift for convergence check
        centroid_shift = np.mean(np.linalg.norm(new_centroids - self.centroids, axis=1))
        
        self.centroids = new_centroids
        return new_centroids, centroid_shift


def create_synthetic_data(n_samples: int, dim: int, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    """Create synthetic clustered data for testing."""
    np.random.seed(seed)
    
    # Create cluster centers
    centers = np.random.randn(n_clusters, dim) * 10
    
    # Generate points around centers
    data = []
    points_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        cluster_data = np.random.randn(points_per_cluster, dim) + centers[i]
        data.append(cluster_data)
    
    # Add remaining points to last cluster
    remaining = n_samples - len(data) * points_per_cluster
    if remaining > 0:
        extra_data = np.random.randn(remaining, dim) + centers[-1]
        data.append(extra_data)
    
    return np.vstack(data).astype(np.float32)


def distributed_faiss_kmeans(data: np.ndarray, k: int, num_workers: int = 4, 
                           max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Perform distributed k-means clustering using FAISS GPU workers.
    
    Args:
        data: Input data (n_samples, n_features)
        k: Number of clusters
        num_workers: Number of GPU workers to use
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        Tuple of (centroids, labels, objective_history)
    """
    n_samples, dim = data.shape
    
    # Split data across workers
    data_splits = np.array_split(data, num_workers)
    
    # Create GPU workers
    logger.info(f"Creating {num_workers} GPU workers...")
    workers = [FAISSGPUKMeansWorker.remote(i, i % torch.cuda.device_count()) 
              for i in range(num_workers)]
    
    # Create coordinator
    coordinator = FAISSDistributedKMeansCoordinator.remote(k, max_iter, tol)
    
    # Initialize centroids using a sample of the data
    sample_size = min(10000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_data = data[sample_indices]
    
    centroids = ray.get(coordinator.initialize_centroids.remote(sample_data))
    
    objective_history = []
    
    logger.info("Starting distributed k-means iterations...")
    
    for iteration in range(max_iter):
        start_time = time.time()
        
        # Assign points to centroids in parallel
        assignment_futures = []
        for i, worker in enumerate(workers):
            future = worker.assign_to_centroids.remote(data_splits[i], centroids, k)
            assignment_futures.append(future)
        
        # Get assignment results
        assignment_results = ray.get(assignment_futures)
        
        # Compute local statistics in parallel
        stats_futures = []
        for i, worker in enumerate(workers):
            distances, assignments = assignment_results[i]
            future = worker.compute_local_statistics.remote(data_splits[i], assignments, k)
            stats_futures.append(future)
        
        # Get statistics results
        stats_results = ray.get(stats_futures)
        
        # Extract centroid sums and counts
        all_centroid_sums = [result[0] for result in stats_results]
        all_cluster_counts = [result[1] for result in stats_results]
        
        # Update centroids
        new_centroids, centroid_shift = ray.get(
            coordinator.update_centroids.remote(all_centroid_sums, all_cluster_counts)
        )
        
        # Compute objective (total within-cluster sum of squares)
        total_objective = sum(np.sum(distances**2) for distances, _ in assignment_results)
        objective_history.append(total_objective)
        
        iteration_time = time.time() - start_time
        
        logger.info(f"Iteration {iteration + 1}/{max_iter}: "
                   f"Objective = {total_objective:.2f}, "
                   f"Centroid shift = {centroid_shift:.6f}, "
                   f"Time = {iteration_time:.2f}s")
        
        # Check for convergence
        if centroid_shift < tol:
            logger.info(f"Converged after {iteration + 1} iterations")
            break
        
        centroids = new_centroids
    
    # Final assignment to get labels
    final_assignment_futures = []
    for i, worker in enumerate(workers):
        future = worker.assign_to_centroids.remote(data_splits[i], centroids, k)
        final_assignment_futures.append(future)
    
    final_assignments = ray.get(final_assignment_futures)
    
    # Combine labels from all workers
    all_labels = []
    for _, assignments in final_assignments:
        all_labels.extend(assignments)
    
    labels = np.array(all_labels)
    
    logger.info("Distributed k-means clustering completed!")
    
    return centroids, labels, objective_history


def main():
    """Main function demonstrating FAISS distributed k-means with GPU support."""
    
    # Initialize Ray
    ray.init()
    
    try:
        # Parameters
        n_samples = 100000
        dim = 128
        k = 10
        num_workers = 4
        
        logger.info(f"Generating {n_samples} samples with {dim} dimensions...")
        
        # Create synthetic data
        data = create_synthetic_data(n_samples, dim, n_clusters=k)
        
        logger.info("Starting distributed FAISS k-means clustering...")
        
        # Perform distributed k-means
        start_time = time.time()
        centroids, labels, objective_history = distributed_faiss_kmeans(
            data, k, num_workers=num_workers, max_iter=50, tol=1e-4
        )
        total_time = time.time() - start_time
        
        # Print results
        logger.info(f"Clustering completed in {total_time:.2f} seconds")
        logger.info(f"Final objective: {objective_history[-1]:.2f}")
        logger.info(f"Number of iterations: {len(objective_history)}")
        
        # Compute cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info("Cluster sizes:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Cluster {label}: {count} points")
        
        # Save results (optional)
        np.save('centroids.npy', centroids)
        np.save('labels.npy', labels)
        np.save('objective_history.npy', objective_history)
        
        logger.info("Results saved to disk")
        
    finally:
        # Clean up Ray
        ray.shutdown()


if __name__ == "__main__":
    main() 