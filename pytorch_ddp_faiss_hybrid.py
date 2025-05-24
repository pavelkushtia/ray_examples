#!/usr/bin/env python3
"""
Hybrid PyTorch DDP + FAISS GPU K-means in Ray Cluster
====================================================

This example demonstrates how to combine PyTorch DistributedDataParallel (DDP)
with FAISS GPU k-means clustering in a Ray cluster for advanced ML workflows.

Use cases:
- Feature clustering during neural network training
- Dynamic clustering-based data sampling
- Representation learning with clustering regularization
- Large-scale clustering with deep learning pipelines
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import ray
import faiss
from typing import List, Tuple, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration."""
    
    def __init__(self, n_samples: int, dim: int, n_clusters: int = 5, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create cluster centers
        self.centers = torch.randn(n_clusters, dim) * 10
        
        # Generate points around centers
        self.data = []
        self.labels = []
        points_per_cluster = n_samples // n_clusters
        
        for i in range(n_clusters):
            cluster_data = torch.randn(points_per_cluster, dim) + self.centers[i]
            cluster_labels = torch.full((points_per_cluster,), i, dtype=torch.long)
            self.data.append(cluster_data)
            self.labels.append(cluster_labels)
        
        # Add remaining points to last cluster
        remaining = n_samples - len(self.data) * points_per_cluster
        if remaining > 0:
            extra_data = torch.randn(remaining, dim) + self.centers[-1]
            extra_labels = torch.full((remaining,), n_clusters - 1, dtype=torch.long)
            self.data.append(extra_data)
            self.labels.append(extra_labels)
        
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FeatureExtractor(nn.Module):
    """Simple neural network for feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


@ray.remote(num_gpus=1)
class FAISSGPUClusteringService:
    """
    Ray actor that provides GPU-accelerated clustering services using FAISS.
    Can be used alongside PyTorch DDP training for hybrid workflows.
    """
    
    def __init__(self, service_id: int, gpu_id: int = 0):
        self.service_id = service_id
        self.gpu_id = gpu_id
        self.gpu_resource = None
        self.current_centroids = None
        self.current_k = None
        
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Initialize FAISS GPU resources."""
        try:
            self.gpu_resource = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = False
            config.device = self.gpu_id
            logger.info(f"FAISS GPU Service {self.service_id}: GPU {self.gpu_id} initialized")
        except Exception as e:
            logger.error(f"FAISS GPU Service {self.service_id}: GPU init failed: {e}")
            self.gpu_resource = None
    
    def cluster_features(self, features: np.ndarray, k: int, max_iter: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform k-means clustering on extracted features.
        
        Args:
            features: Feature vectors to cluster (n_samples, dim)
            k: Number of clusters
            max_iter: Maximum iterations for k-means
            
        Returns:
            Tuple of (centroids, cluster_assignments)
        """
        try:
            n_samples, dim = features.shape
            
            # Initialize clustering
            clustering = faiss.Clustering(dim, k)
            clustering.verbose = False
            clustering.niter = max_iter
            clustering.nredo = 1
            clustering.seed = 42
            
            # Create index
            if self.gpu_resource is not None:
                config = faiss.GpuIndexFlatConfig()
                config.useFloat16 = False
                config.device = self.gpu_id
                index = faiss.GpuIndexFlatL2(self.gpu_resource, dim, config)
            else:
                index = faiss.IndexFlatL2(dim)
            
            # Perform clustering
            clustering.train(features.astype(np.float32), index)
            
            # Get centroids
            centroids = faiss.vector_to_array(clustering.centroids).reshape(k, dim)
            
            # Assign points to clusters
            distances, assignments = index.search(features.astype(np.float32), 1)
            assignments = assignments.flatten()
            
            self.current_centroids = centroids
            self.current_k = k
            
            logger.info(f"FAISS Service {self.service_id}: Clustered {n_samples} features into {k} clusters")
            
            return centroids, assignments
            
        except Exception as e:
            logger.error(f"FAISS Service {self.service_id}: Clustering failed: {e}")
            raise
    
    def assign_to_existing_clusters(self, features: np.ndarray) -> np.ndarray:
        """
        Assign new features to existing cluster centroids.
        
        Args:
            features: New feature vectors to assign
            
        Returns:
            Cluster assignments
        """
        if self.current_centroids is None:
            raise ValueError("No existing clusters. Run cluster_features first.")
        
        try:
            dim = features.shape[1]
            
            # Create index with current centroids
            if self.gpu_resource is not None:
                config = faiss.GpuIndexFlatConfig()
                config.useFloat16 = False
                config.device = self.gpu_id
                index = faiss.GpuIndexFlatL2(self.gpu_resource, dim, config)
            else:
                index = faiss.IndexFlatL2(dim)
            
            # Add centroids to index
            index.add(self.current_centroids.astype(np.float32))
            
            # Assign features
            distances, assignments = index.search(features.astype(np.float32), 1)
            
            return assignments.flatten()
            
        except Exception as e:
            logger.error(f"FAISS Service {self.service_id}: Assignment failed: {e}")
            raise


@ray.remote(num_gpus=1)
class PyTorchDDPTrainer:
    """
    Ray actor that handles PyTorch DDP training with FAISS clustering integration.
    """
    
    def __init__(self, rank: int, world_size: int, gpu_id: int, clustering_service):
        self.rank = rank
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.clustering_service = clustering_service
        self.device = None
        self.model = None
        self.optimizer = None
        
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        try:
            # Set CUDA device
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device(f'cuda:{self.gpu_id}')
            
            # Initialize process group
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = str(self.rank)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            
            # Note: In a real Ray cluster, you'd need proper distributed setup
            # This is simplified for demonstration
            
            logger.info(f"DDP Trainer {self.rank}: Device setup complete")
            
        except Exception as e:
            logger.error(f"DDP Trainer {self.rank}: Setup failed: {e}")
            raise
    
    def setup_model(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Setup the neural network model with DDP."""
        try:
            # Create model
            self.model = FeatureExtractor(input_dim, hidden_dim, output_dim)
            self.model = self.model.to(self.device)
            
            # Wrap with DDP (simplified - in real setup you'd need proper process group)
            # self.model = DDP(self.model, device_ids=[self.gpu_id])
            
            # Setup optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            logger.info(f"DDP Trainer {self.rank}: Model setup complete")
            
        except Exception as e:
            logger.error(f"DDP Trainer {self.rank}: Model setup failed: {e}")
            raise
    
    def train_with_clustering(self, dataset: Dataset, batch_size: int, 
                            num_epochs: int, k_clusters: int) -> Dict[str, Any]:
        """
        Train model with periodic clustering of learned features.
        
        Args:
            dataset: Training dataset
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            k_clusters: Number of clusters for feature clustering
            
        Returns:
            Training metrics and clustering results
        """
        try:
            # Create data loader
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, 
                                       rank=self.rank, shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
            
            results = {
                'losses': [],
                'cluster_metrics': [],
                'epoch_times': []
            }
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                sampler.set_epoch(epoch)
                
                # Training phase
                self.model.train()
                epoch_loss = 0.0
                all_features = []
                all_labels = []
                
                for batch_idx, (data, labels) in enumerate(dataloader):
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    features = self.model(data)
                    
                    # Simple contrastive loss (for demonstration)
                    loss = self._compute_contrastive_loss(features, labels)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Collect features for clustering
                    all_features.append(features.detach().cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                
                # Clustering phase (every few epochs)
                if (epoch + 1) % 5 == 0:
                    features_array = np.vstack(all_features)
                    labels_array = np.concatenate(all_labels)
                    
                    # Perform clustering using FAISS service
                    centroids, cluster_assignments = ray.get(
                        self.clustering_service.cluster_features.remote(
                            features_array, k_clusters
                        )
                    )
                    
                    # Compute clustering metrics
                    cluster_metrics = self._compute_clustering_metrics(
                        labels_array, cluster_assignments
                    )
                    results['cluster_metrics'].append(cluster_metrics)
                    
                    logger.info(f"DDP Trainer {self.rank}, Epoch {epoch + 1}: "
                              f"Loss = {epoch_loss:.4f}, "
                              f"Clustering ARI = {cluster_metrics['ari']:.4f}")
                
                epoch_time = time.time() - epoch_start
                results['losses'].append(epoch_loss)
                results['epoch_times'].append(epoch_time)
            
            return results
            
        except Exception as e:
            logger.error(f"DDP Trainer {self.rank}: Training failed: {e}")
            raise
    
    def _compute_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Simple contrastive loss for demonstration."""
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(features, features)
        
        # Create positive and negative masks
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = 1.0 - positive_mask
        
        # Remove diagonal (self-distances)
        mask = torch.eye(len(features), device=features.device)
        positive_mask = positive_mask * (1.0 - mask)
        negative_mask = negative_mask * (1.0 - mask)
        
        # Contrastive loss
        positive_loss = positive_mask * distances.pow(2)
        negative_loss = negative_mask * torch.clamp(2.0 - distances, min=0).pow(2)
        
        loss = (positive_loss.sum() + negative_loss.sum()) / (positive_mask.sum() + negative_mask.sum() + 1e-8)
        
        return loss
    
    def _compute_clustering_metrics(self, true_labels: np.ndarray, 
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering evaluation metrics."""
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            
            return {'ari': ari, 'nmi': nmi}
            
        except ImportError:
            # Fallback if sklearn not available
            return {'ari': 0.0, 'nmi': 0.0}


def main():
    """Main function demonstrating hybrid PyTorch DDP + FAISS GPU clustering."""
    
    # Initialize Ray
    ray.init()
    
    try:
        # Parameters
        n_samples = 50000
        input_dim = 100
        hidden_dim = 256
        output_dim = 64
        k_clusters = 10
        num_workers = 2
        batch_size = 256
        num_epochs = 20
        
        logger.info("Setting up hybrid PyTorch DDP + FAISS clustering...")
        
        # Create dataset
        dataset = SyntheticDataset(n_samples, input_dim, n_clusters=k_clusters)
        
        # Create FAISS clustering service
        clustering_service = FAISSGPUClusteringService.remote(0, 0)
        
        # Create DDP trainers
        trainers = []
        for rank in range(num_workers):
            trainer = PyTorchDDPTrainer.remote(
                rank=rank,
                world_size=num_workers,
                gpu_id=rank % torch.cuda.device_count(),
                clustering_service=clustering_service
            )
            trainers.append(trainer)
        
        # Setup models
        setup_futures = []
        for trainer in trainers:
            future = trainer.setup_model.remote(input_dim, hidden_dim, output_dim)
            setup_futures.append(future)
        
        ray.get(setup_futures)
        
        logger.info("Starting hybrid training with periodic clustering...")
        
        # Start training
        start_time = time.time()
        training_futures = []
        
        for trainer in trainers:
            future = trainer.train_with_clustering.remote(
                dataset, batch_size, num_epochs, k_clusters
            )
            training_futures.append(future)
        
        # Get training results
        training_results = ray.get(training_futures)
        
        total_time = time.time() - start_time
        
        # Aggregate and display results
        logger.info(f"Hybrid training completed in {total_time:.2f} seconds")
        
        for rank, results in enumerate(training_results):
            final_loss = results['losses'][-1] if results['losses'] else 0.0
            avg_epoch_time = np.mean(results['epoch_times'])
            
            logger.info(f"Trainer {rank}: Final loss = {final_loss:.4f}, "
                       f"Avg epoch time = {avg_epoch_time:.2f}s")
            
            if results['cluster_metrics']:
                final_ari = results['cluster_metrics'][-1]['ari']
                final_nmi = results['cluster_metrics'][-1]['nmi']
                logger.info(f"Trainer {rank}: Final clustering ARI = {final_ari:.4f}, "
                           f"NMI = {final_nmi:.4f}")
        
        logger.info("Hybrid PyTorch DDP + FAISS clustering completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Clean up Ray
        ray.shutdown()


if __name__ == "__main__":
    main() 