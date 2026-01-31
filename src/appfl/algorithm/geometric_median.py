"""
Geometric Median (L1-median) Implementation for Robust Federated Learning

This module implements the geometric median using the smoothed Weiszfeld algorithm.
The geometric median is a robust aggregation method that can handle up to 50% corrupted/outlier data.

Reference:
    Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2022). 
    Robust Aggregation for Federated Learning. 
    IEEE Transactions on Signal Processing, 70, 1142-1154.
    
The geometric median minimizes the sum of L2 distances to all points:
    argmin_y Σᵢ ||points[i] - y||₂
    
For weighted version:
    argmin_y Σᵢ weights[i] * ||points[i] - y||₂
"""

import torch
from collections import OrderedDict
from typing import List, Dict, Optional, Union
import copy


def compute_geometric_median(
    points: List[Dict[str, torch.Tensor]],
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-10
) -> Dict[str, torch.Tensor]:
    """
    Compute the geometric median of model parameters using Weiszfeld's algorithm.
    
    The geometric median minimizes the weighted sum of L2 distances:
        argmin_y Σᵢ weights[i] * ||points[i] - y||₂
    
    Args:
        points: List of state_dicts from different clients
                Each state_dict contains {param_name: tensor}
        weights: Optional weights for each point (client). If None, uniform weights are used.
                 Shape: (num_clients,)
        eps: Smoothing parameter for numerical stability (prevents division by zero)
        maxiter: Maximum number of iterations for the algorithm
        ftol: Tolerance for convergence (stop when objective function change < ftol)
    
    Returns:
        state_dict: Geometric median of the input points
        
    Example:
        >>> client_models = [model1.state_dict(), model2.state_dict(), model3.state_dict()]
        >>> median_model = compute_geometric_median(client_models)
    """
    
    num_points = len(points)
    
    if num_points == 0:
        raise ValueError("Cannot compute geometric median of empty list")
    
    if num_points == 1:
        return copy.deepcopy(points[0])
    
    # Set uniform weights if not provided
    if weights is None:
        weights = torch.ones(num_points, device=next(iter(points[0].values())).device)
        weights = weights / weights.sum()
    else:
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
    
    # Get device from first parameter
    device = next(iter(points[0].values())).device
    weights = weights.to(device)
    
    # Initialize median as weighted average (good starting point)
    median = OrderedDict()
    for key in points[0].keys():
        median[key] = torch.zeros_like(points[0][key], device=device)
        for i, point in enumerate(points):
            median[key] += weights[i] * point[key].to(device)
    
    # Weiszfeld's algorithm with smoothing
    obj_val_prev = float('inf')
    
    for iteration in range(maxiter):
        # Compute distances from current median to each point
        # Distance for point i: ||point[i] - median||₂ = sqrt(Σₖ ||point[i][k] - median[k]||²)
        distances = torch.zeros(num_points, device=device)
        for i, point in enumerate(points):
            dist_squared = 0.0
            for key in point.keys():
                diff = point[key].to(device) - median[key]
                dist_squared += torch.sum(diff ** 2)
            # Add eps² for smoothing (smoothed Weiszfeld)
            distances[i] = torch.sqrt(dist_squared + eps**2)
        
        # Compute objective function value (weighted sum of distances)
        obj_val = torch.sum(weights * distances).item()
        
        # Check convergence
        if abs(obj_val - obj_val_prev) < ftol:
            break
        obj_val_prev = obj_val
        
        # Compute Weiszfeld weights
        # For smoothed Weiszfeld: wᵢ = weight[i] / distance[i]
        # Then normalize so they sum to 1
        inv_distances = 1.0 / distances
        weiszfeld_weights = weights * inv_distances
        weiszfeld_weights = weiszfeld_weights / weiszfeld_weights.sum()
        
        # Update median: weighted average with Weiszfeld weights
        new_median = OrderedDict()
        for key in points[0].keys():
            new_median[key] = torch.zeros_like(median[key], device=device)
            for i, point in enumerate(points):
                new_median[key] += weiszfeld_weights[i] * point[key].to(device)
        
        median = new_median
    
    return median


def compute_geometric_median_per_layer(
    points: List[Dict[str, torch.Tensor]],
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-10
) -> Dict[str, torch.Tensor]:
    """
    Compute geometric median separately for each layer/parameter.
    
    This is computationally more efficient and can be more robust when different
    layers have different scales or corruption patterns.
    
    Args:
        points: List of state_dicts from different clients
        weights: Optional weights for each point (client)
        eps: Smoothing parameter
        maxiter: Maximum iterations per layer
        ftol: Convergence tolerance per layer
    
    Returns:
        state_dict: Per-layer geometric median
    """
    
    num_points = len(points)
    
    if num_points == 0:
        raise ValueError("Cannot compute geometric median of empty list")
    
    if num_points == 1:
        return copy.deepcopy(points[0])
    
    # Set uniform weights if not provided
    if weights is None:
        device = next(iter(points[0].values())).device
        weights = torch.ones(num_points, device=device) / num_points
    else:
        weights = weights / weights.sum()
    
    device = next(iter(points[0].values())).device
    weights = weights.to(device)
    
    median = OrderedDict()
    
    # Compute geometric median for each parameter independently
    for key in points[0].keys():
        # Stack all client parameters for this layer
        layer_points = [point[key].to(device) for point in points]
        
        # Initialize with weighted average
        median[key] = torch.zeros_like(layer_points[0], device=device)
        for i, layer_point in enumerate(layer_points):
            median[key] += weights[i] * layer_point
        
        # Weiszfeld's algorithm for this layer
        obj_val_prev = float('inf')
        
        for iteration in range(maxiter):
            # Compute distances
            distances = torch.zeros(num_points, device=device)
            for i, layer_point in enumerate(layer_points):
                diff = layer_point - median[key]
                distances[i] = torch.sqrt(torch.sum(diff ** 2) + eps**2)
            
            # Objective value
            obj_val = torch.sum(weights * distances).item()
            
            # Check convergence
            if abs(obj_val - obj_val_prev) < ftol:
                break
            obj_val_prev = obj_val
            
            # Weiszfeld weights
            inv_distances = 1.0 / distances
            weiszfeld_weights = weights * inv_distances
            weiszfeld_weights = weiszfeld_weights / weiszfeld_weights.sum()
            
            # Update median
            new_median = torch.zeros_like(median[key], device=device)
            for i, layer_point in enumerate(layer_points):
                new_median += weiszfeld_weights[i] * layer_point
            
            median[key] = new_median
    
    return median


def robust_aggregate(
    models_state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[torch.Tensor] = None,
    method: str = "geometric_median",
    per_layer: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    High-level function for robust aggregation in federated learning.
    
    Args:
        models_state_dicts: List of model state_dicts from clients
        weights: Optional client weights
        method: Aggregation method. Currently supports:
                - "geometric_median": Geometric median (L1-median)
                - "mean": Simple weighted average (not robust)
        per_layer: Whether to compute geometric median per layer (more efficient)
        **kwargs: Additional arguments passed to the aggregation function
    
    Returns:
        Aggregated state_dict
    """
    
    if method == "geometric_median":
        if per_layer:
            return compute_geometric_median_per_layer(models_state_dicts, weights, **kwargs)
        else:
            return compute_geometric_median(models_state_dicts, weights, **kwargs)
    elif method == "mean":
        # Fallback to simple averaging
        return _weighted_average(models_state_dicts, weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _weighted_average(
    points: List[Dict[str, torch.Tensor]],
    weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute weighted average of model parameters (non-robust baseline)"""
    
    num_points = len(points)
    device = next(iter(points[0].values())).device
    
    if weights is None:
        weights = torch.ones(num_points, device=device) / num_points
    else:
        weights = weights / weights.sum()
        weights = weights.to(device)
    
    average = OrderedDict()
    for key in points[0].keys():
        average[key] = torch.zeros_like(points[0][key], device=device)
        for i, point in enumerate(points):
            average[key] += weights[i] * point[key].to(device)
    
    return average
