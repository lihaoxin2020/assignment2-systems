#!/usr/bin/env python3
"""
Benchmarking script for cs336_basics models.

This script performs end-to-end benchmarking of forward and backward passes
for the BasicsTransformerLM model with preset configurations for different model sizes.
"""

import argparse
import timeit
import sys
import os
from typing import Dict, Any, Tuple
import logging
import math
from einops import einsum
from jaxtyping import Float, Bool, Int
from torch import Tensor

from cs336_basics.nn_utils import softmax
import torch.cuda.nvtx as nvtx

# Force line buffering for better nsys compatibility
os.environ['PYTHONUNBUFFERED'] = '1'

print("Loading PyTorch...", flush=True)
try:
    import torch
    import torch.nn as nn
    print(f"PyTorch loaded successfully. Version: {torch.__version__}", flush=True)
except Exception as e:
    print(f"Error loading PyTorch: {e}", flush=True)
    sys.exit(1)

# Import the model from cs336_basics
print("Loading cs336_basics model...", flush=True)
try:
    from cs336_basics.model import BasicsTransformerLM
    print("cs336_basics model loaded successfully", flush=True)
except ImportError as e:
    print(f"Error: Could not import cs336_basics.model: {e}", flush=True)
    print("Make sure cs336-basics is installed.", flush=True)
    sys.exit(1)

# Set up logging with explicit flushing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force logging to flush immediately
class FlushingHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger.handlers = [FlushingHandler(sys.stdout)]

# Preset model configurations
MODEL_CONFIGS = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 50257,
        'context_length': 1024,
        'rope_theta': 10000.0
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16,
        'vocab_size': 50257,
        'context_length': 1024,
        'rope_theta': 10000.0
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20,
        'vocab_size': 50257,
        'context_length': 1024,
        'rope_theta': 10000.0
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25,
        'vocab_size': 50257,
        'context_length': 1024,
        'rope_theta': 10000.0
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32,
        'vocab_size': 50257,
        'context_length': 1024,
        'rope_theta': 10000.0
    }
}


def get_model_config(size: str) -> Dict[str, Any]:
    """
    Get preset model configuration for a given size.
    
    Args:
        size: Model size ('small', 'medium', 'large', 'xl', '2.7B')
        
    Returns:
        Dictionary with model configuration
        
    Raises:
        ValueError: If size is not recognized
    """
    if size not in MODEL_CONFIGS:
        available_sizes = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model size '{size}'. Available sizes: {available_sizes}")
    
    return MODEL_CONFIGS[size].copy()


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return output


def create_model(hyperparameters: Dict[str, Any]) -> BasicsTransformerLM:
    """
    Initialize a BasicsTransformerLM model with given hyperparameters.
    
    Args:
        hyperparameters: Dictionary containing model configuration
        
    Returns:
        Initialized BasicsTransformerLM model
    """
    print("Creating model...", flush=True)
    
    print("Initializing BasicsTransformerLM...", flush=True)
    model = BasicsTransformerLM(
        vocab_size=hyperparameters['vocab_size'],
        context_length=hyperparameters['context_length'],
        d_model=hyperparameters['d_model'],
        num_layers=hyperparameters['num_layers'],
        num_heads=hyperparameters['num_heads'],
        d_ff=hyperparameters['d_ff'],
        rope_theta=hyperparameters.get('rope_theta', 10000.0)
    )
    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    print("Model initialization complete, checking device...", flush=True)
    
    # Check CUDA availability first
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
        print(f"CUDA device name: {torch.cuda.get_device_name()}", flush=True)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Moving model to device: {device}", flush=True)
    
    # For nsys profiling, try CPU fallback if CUDA move hangs
    if device.type == 'cuda':
        try:
            # Pre-warm CUDA context
            torch.cuda.empty_cache()
            dummy = torch.randn(1, device=device)
            del dummy
            torch.cuda.empty_cache()
            
            # Move model to CUDA
            model = model.to(device)
            print("Model successfully moved to CUDA", flush=True)
        except Exception as e:
            print(f"Error moving model to CUDA: {e}, falling back to CPU", flush=True)
            device = torch.device('cpu')
            model = model.to(device)
    else:
        model = model.to(device)
        print("Model successfully moved to device", flush=True)
    
    print("Getting model parameter count...", flush=True)
    try:
        param_count = model.get_num_params()
        print(f"Parameter count: {param_count}", flush=True)
    except Exception as e:
        print(f"Error getting parameter count: {e}", flush=True)
        param_count = 0
    
    logger.info(f"Model created with {param_count/1e6:.2f}M parameters on device: {device}")
    print(f"Model ready on device: {device}", flush=True)
    return model


def create_random_batch(batch_size: int, sequence_length: int, vocab_size: int) -> torch.Tensor:
    """
    Generate a random batch of data for the model.
    
    Args:
        batch_size: Number of sequences in the batch
        sequence_length: Length of each sequence
        vocab_size: Size of the vocabulary
        
    Returns:
        Random batch of token IDs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)


def run_forward_pass(model: BasicsTransformerLM, input_batch: torch.Tensor) -> torch.Tensor:
    """
    Run a single forward pass.
    
    Args:
        model: The model to benchmark
        input_batch: Input batch of token IDs
        
    Returns:
        Model output
    """
    output = model(input_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return output


def run_forward_and_backward_pass(model: BasicsTransformerLM, input_batch: torch.Tensor, 
                                  criterion: nn.Module, target_batch: torch.Tensor) -> float:
    """
    Run a single forward and backward pass.
    
    Args:
        model: The model to benchmark
        input_batch: Input batch of token IDs
        criterion: Loss function
        target_batch: Target batch for loss computation
        
    Returns:
        Loss value
    """
    # Clear gradients
    model.zero_grad()
    
    # Forward pass
    output = model(input_batch)
    
    # Compute loss (shift for language modeling)
    loss = criterion(output[:, :-1, :].contiguous().view(-1, output.size(-1)), 
                    target_batch[:, 1:].contiguous().view(-1))
    
    # Backward pass
    loss.backward()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    return loss.item()


def benchmark_model(model: BasicsTransformerLM, 
                   input_batch: torch.Tensor,
                   target_batch: torch.Tensor,
                   warmup_steps: int = 10,
                   benchmark_steps: int = 100,
                   include_backward: bool = True) -> Dict[str, float]:
    """
    Benchmark the model with the given configuration.
    
    Args:
        model: The model to benchmark
        input_batch: Input batch for benchmarking
        target_batch: Target batch for loss computation (used only if include_backward=True)
        warmup_steps: Number of warmup steps before timing
        benchmark_steps: Number of steps to time
        include_backward: Whether to include backward pass in timing
        
    Returns:
        Dictionary with timing results
    """
    print(f"Starting benchmark with {warmup_steps} warmup steps and {benchmark_steps} benchmark steps", flush=True)
    
    # Prepare loss function if needed
    criterion = None
    if include_backward:
        criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Starting warmup: {warmup_steps} steps")
    print(f"Warmup phase starting...", flush=True)
    
    # Warmup phase
    for i in range(warmup_steps):
        # if i % max(1, warmup_steps // 4) == 0:
        #     print(f"Warmup step {i+1}/{warmup_steps}", flush=True)
        if include_backward:
            run_forward_and_backward_pass(model, input_batch, criterion, target_batch)
        else:
            run_forward_pass(model, input_batch)
    
    logger.info(f"Warmup complete. Starting benchmark: {benchmark_steps} steps")
    print(f"Benchmark phase starting...", flush=True)
    
    # Benchmarking phase using timeit.default_timer() for highest resolution
    def single_step():
        if include_backward:
            return run_forward_and_backward_pass(model, input_batch, criterion, target_batch)
        else:
            run_forward_pass(model, input_batch)
            return 0.0
    
    # Time the benchmark steps
    start_time = timeit.default_timer()
    total_loss = 0.0
    
    for i in range(benchmark_steps):
        # if i % max(1, benchmark_steps // 10) == 0:
        #     print(f"Benchmark step {i+1}/{benchmark_steps}", flush=True)
        loss = single_step()
        total_loss += loss
    
    end_time = timeit.default_timer()
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / benchmark_steps
    avg_loss = total_loss / benchmark_steps if include_backward else None
    
    print("Benchmark completed!", flush=True)
    
    results = {
        'total_time': total_time,
        'avg_time_per_step': avg_time_per_step,
        'steps_per_second': benchmark_steps / total_time,
        'benchmark_steps': benchmark_steps,
        'include_backward': include_backward
    }
    
    if include_backward:
        results['avg_loss'] = avg_loss
    
    return results


def print_results(results: Dict[str, float], hyperparameters: Dict[str, Any], 
                 batch_size: int, sequence_length: int, model_size: str = None):
    """
    Print benchmarking results in a formatted way.
    
    Args:
        results: Benchmarking results
        hyperparameters: Model hyperparameters
        batch_size: Batch size used
        sequence_length: Sequence length used
        model_size: Model size name if using preset
    """
    print("\n" + "="*60, flush=True)
    print("BENCHMARKING RESULTS", flush=True)
    print("="*60, flush=True)
    
    if model_size:
        print(f"Model Size: {model_size}", flush=True)
        
    print(f"Model Configuration:", flush=True)
    print(f"  - Layers: {hyperparameters['num_layers']}", flush=True)
    print(f"  - Hidden size (d_model): {hyperparameters['d_model']}", flush=True)
    print(f"  - Attention heads: {hyperparameters['num_heads']}", flush=True)
    print(f"  - Feed-forward size: {hyperparameters['d_ff']}", flush=True)
    print(f"  - Vocabulary size: {hyperparameters['vocab_size']}", flush=True)
    print(f"  - Context length: {hyperparameters['context_length']}", flush=True)
    
    print(f"\nBatch Configuration:", flush=True)
    print(f"  - Batch size: {batch_size}", flush=True)
    print(f"  - Sequence length: {sequence_length}", flush=True)
    
    print(f"\nTiming Results:", flush=True)
    print(f"  - Total time: {results['total_time']:.4f} seconds", flush=True)
    print(f"  - Average time per step: {results['avg_time_per_step']*1000:.2f} ms", flush=True)
    print(f"  - Steps per second: {results['steps_per_second']:.2f}", flush=True)
    print(f"  - Benchmark steps: {results['benchmark_steps']}", flush=True)
    print(f"  - Include backward pass: {results['include_backward']}", flush=True)
    
    if results['include_backward'] and 'avg_loss' in results:
        print(f"  - Average loss: {results['avg_loss']:.4f}", flush=True)
    
    # Calculate throughput
    tokens_per_step = batch_size * sequence_length
    tokens_per_second = tokens_per_step * results['steps_per_second']
    print(f"  - Tokens per second: {tokens_per_second:.0f}", flush=True)
    
    print("="*60, flush=True)


def run_benchmark(model_size: str = None,
                 num_layers: int = None, 
                 d_model: int = None,
                 num_heads: int = None,
                 d_ff: int = None,
                 vocab_size: int = 50257,
                 context_length: int = 1024,
                 rope_theta: float = 10000.0,
                 batch_size: int = 4,
                 sequence_length: int = 512,
                 warmup_steps: int = 10,
                 benchmark_steps: int = 100,
                 include_backward: bool = True,
                 print_results_flag: bool = True) -> Dict[str, float]:
    """
    Convenience function to run a benchmark with given parameters.
    
    This is a programmatic interface to the benchmarking functionality.
    
    Args:
        model_size: Preset model size ('small', 'medium', 'large', 'xl', '2.7B')
        num_layers: Number of transformer layers (overrides preset if provided)
        d_model: Model dimension (overrides preset if provided)
        num_heads: Number of attention heads (overrides preset if provided)
        d_ff: Feed-forward dimension (overrides preset if provided)
        vocab_size: Vocabulary size
        context_length: Maximum context length
        rope_theta: RoPE theta parameter
        batch_size: Batch size for benchmarking
        sequence_length: Sequence length for benchmarking
        warmup_steps: Number of warmup steps
        benchmark_steps: Number of benchmark steps
        include_backward: Whether to include backward pass
        print_results_flag: Whether to print formatted results
        
    Returns:
        Dictionary with timing results
    """
    print(f"Starting benchmark with model_size={model_size}", flush=True)
    
    # Get hyperparameters
    if model_size:
        print(f"Getting config for model size: {model_size}", flush=True)
        hyperparameters = get_model_config(model_size)
        # Override with any explicitly provided parameters
        if num_layers is not None:
            hyperparameters['num_layers'] = num_layers
        if d_model is not None:
            hyperparameters['d_model'] = d_model
        if num_heads is not None:
            hyperparameters['num_heads'] = num_heads
        if d_ff is not None:
            hyperparameters['d_ff'] = d_ff
        hyperparameters['vocab_size'] = vocab_size
        hyperparameters['context_length'] = context_length
        hyperparameters['rope_theta'] = rope_theta
    else:
        # Manual configuration
        if any(param is None for param in [num_layers, d_model, num_heads, d_ff]):
            raise ValueError("If model_size is not provided, num_layers, d_model, num_heads, and d_ff must be specified")
        
        hyperparameters = {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'rope_theta': rope_theta
        }
    
    print(f"Hyperparameters: {hyperparameters}", flush=True)
    
    # Validate arguments
    if hyperparameters['d_model'] % hyperparameters['num_heads'] != 0:
        raise ValueError(f"d_model ({hyperparameters['d_model']}) must be divisible by num_heads ({hyperparameters['num_heads']})")
    
    if sequence_length > hyperparameters['context_length']:
        raise ValueError(f"sequence_length ({sequence_length}) cannot exceed context_length ({hyperparameters['context_length']})")
    
    # Create model
    print("Creating model...", flush=True)
    model = create_model(hyperparameters)
    
    # Create random batch
    print("Creating random batches...", flush=True)
    input_batch = create_random_batch(batch_size, sequence_length, hyperparameters['vocab_size'])
    target_batch = create_random_batch(batch_size, sequence_length, hyperparameters['vocab_size'])
    
    # Run benchmark
    print("Running benchmark...", flush=True)
    results = benchmark_model(
        model=model,
        input_batch=input_batch,
        target_batch=target_batch,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        include_backward=include_backward
    )
    
    # Print results if requested
    if print_results_flag:
        print_results(results, hyperparameters, batch_size, sequence_length, model_size)
    
    print("Benchmark completed successfully!", flush=True)
    return results


def list_model_sizes():
    """Print available preset model sizes and their configurations."""
    print("\nAvailable preset model sizes:")
    print("="*80)
    for size, config in MODEL_CONFIGS.items():
        print(f"{size:>8}: d_model={config['d_model']:>4}, d_ff={config['d_ff']:>5}, "
              f"num_layers={config['num_layers']:>2}, num_heads={config['num_heads']:>2}")
    print("="*80)


def main():
    print("Script starting...", flush=True)
    
    parser = argparse.ArgumentParser(description='Benchmark cs336_basics BasicsTransformerLM model')
    
    # Model configuration - either preset or manual
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model_size', type=str, choices=list(MODEL_CONFIGS.keys()),
                           help='Use preset model configuration (small, medium, large, xl, 2.7B)')
    
    # Manual model hyperparameters (used if --model_size not provided)
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, help='Feed-forward dimension')
    
    # Other model parameters
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024, help='Maximum context length')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Batch configuration
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=512, help='Sequence length')
    
    # Benchmarking configuration
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--benchmark_steps', type=int, default=100, help='Number of benchmark steps')
    parser.add_argument('--forward_only', action='store_true', 
                       help='Only benchmark forward pass (default: benchmark both forward and backward)')
    
    # Utility options
    parser.add_argument('--list_sizes', action='store_true',
                       help='List available preset model sizes and exit')
    
    print("Parsing arguments...", flush=True)
    args = parser.parse_args()
    
    if args.list_sizes:
        list_model_sizes()
        return
    
    print("Starting benchmark with parsed arguments...", flush=True)
    
    # Use the convenience function
    try:
        run_benchmark(
            model_size=args.model_size,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            rope_theta=args.rope_theta,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            include_backward=not args.forward_only,
            print_results_flag=True
        )
    except Exception as e:
        print(f"Error during benchmarking: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Script completed successfully!", flush=True)


if __name__ == '__main__':
    main()