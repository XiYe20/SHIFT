"""
Memory Optimization Mixin for existing trainers
Adds DeepSpeed, Flash Attention, and xFormers support with minimal changes
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import deepspeed
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# Add these lines after the imports and before the class definition
FLASH_ATTENTION_AVAILABLE = False
PYTORCH_FLASH_AVAILABLE = False

try:
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        PYTORCH_FLASH_AVAILABLE = True
        FLASH_ATTENTION_AVAILABLE = True
except:
    pass

class MemoryOptimizationMixin:
    """
    Mixin class to add memory optimizations to existing trainers
    """
    
    def setup_memory_optimizations(self):
        """Setup all memory optimizations - call this in __init__"""
        self._setup_flash_attention()
        self._setup_xformers()
        self._print_memory_optimizations()

        # Print DeepSpeed status
        use_deepspeed = getattr(self.config, 'use_deepspeed', False)
        print(f"DeepSpeed ZeRO: {'✓ Enabled' if use_deepspeed else '✗ Disabled'}")
    
    def _setup_flash_attention(self):
        """Enable Flash Attention where possible and configured"""
        # Check config flag first
        if not getattr(self.config, 'use_flash_attention', True):
            print("Flash Attention disabled by config")
            return
        
        # Use PyTorch's native Flash Attention (available in PyTorch 2.0+)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
            print("✓ Flash Attention enabled via PyTorch native SDPA")
            return
        except Exception as e:
            print(f"PyTorch Flash Attention setup failed: {e}")
        
        # Fallback: Try external flash-attn if available
        try:
            from flash_attn import flash_attn_func
            print("✓ External Flash Attention available as fallback")
        except ImportError:
            print("External Flash Attention not available - using PyTorch native only")

    def _setup_xformers(self):
        """Enable xFormers optimizations where configured"""
        # Check config flag first
        if not getattr(self.config, 'use_xformers', True):
            print("xFormers disabled by config")
            return
            
        if not XFORMERS_AVAILABLE:
            print("xFormers not available - install with: pip install xformers")
            return
            
        # Apply xFormers to UNet attention layers
        def enable_xformers_recursive(module):
            for name, child in module.named_children():
                if hasattr(child, 'enable_xformers_memory_efficient_attention'):
                    try:
                        child.enable_xformers_memory_efficient_attention()
                        print(f"✓ xFormers enabled for {name}")
                    except Exception as e:
                        print(f"xFormers failed for {name}: {e}")
                else:
                    enable_xformers_recursive(child)
        
        if hasattr(self, 'soc_pipeline') and hasattr(self.soc_pipeline, 'unet'):
            enable_xformers_recursive(self.soc_pipeline.unet)
        
        if hasattr(self, 'unet'):
            enable_xformers_recursive(self.unet)
    
    def _print_memory_optimizations(self):
        """Print enabled optimizations"""
        print("=== Memory Optimizations Status ===")
        print(f"Flash Attention: {'✓' if FLASH_ATTENTION_AVAILABLE else '✗'}")
        print(f"xFormers: {'✓' if XFORMERS_AVAILABLE else '✗'}")
        print(f"DeepSpeed: {'✓' if deepspeed else '✗'}")
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Get DeepSpeed configuration for ZeRO-3"""
        # Check if DeepSpeed is enabled
        if not getattr(self.config, 'use_deepspeed', False):
            return None
            
        # Get DeepSpeed stage from config
        stage = getattr(self.config, 'deepspeed_stage', 3)
        cpu_offload = getattr(self.config, 'cpu_offload', True)
        
        return {
            # ZeRO Configuration
            "zero_optimization": {
                "stage": stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
                "sub_group_size": 1e9,
                "offload_optimizer": {
                    "device": "cpu" if cpu_offload else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if cpu_offload else "none", 
                    "pin_memory": True
                }
            },
            
            # Rest remains the same...
            "gradient_clipping": getattr(self.config, 'gradient_clip', 1.0),
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": True,
                "synchronize_checkpoint_boundary": True
            },
            
            "bf16": {
                "enabled": getattr(self.config, 'precision', 'bf16') == 'bf16'
            },
            "fp16": {
                "enabled": getattr(self.config, 'precision', 'bf16') == 'fp16'
            }
        }
    
    def estimate_memory_usage(self):
        """Estimate memory usage with current settings"""
        if hasattr(self, 'unet'):
            try:
                estimate_zero3_model_states_mem_needs_all_live(
                    self.unet, 
                    num_gpus_per_node=torch.cuda.device_count(),
                    num_nodes=1
                )
            except:
                print("Memory estimation not available")