"""
DeepSpeed strategy configuration for PyTorch Lightning
"""

from pytorch_lightning.strategies import DeepSpeedStrategy
from memory_optimization_mixin import MemoryOptimizationMixin

def create_deepspeed_strategy(config):
    """Create DeepSpeed strategy with ZeRO-3 configuration"""
    
    # Check if DeepSpeed is enabled in config
    if not getattr(config, 'use_deepspeed', False):
        print("DeepSpeed disabled by config, using DDP instead")
        return "ddp"
    
    # Create a temporary mixin instance to get the config
    class TempMixin(MemoryOptimizationMixin):
        def __init__(self, config):
            self.config = config
    
    temp = TempMixin(config)
    deepspeed_config = temp.get_deepspeed_config()
    
    if deepspeed_config is None:
        return "ddp"
    
    stage = getattr(config, 'deepspeed_stage', 3)
    cpu_offload = getattr(config, 'cpu_offload', True)
    
    return DeepSpeedStrategy(
        stage=stage,
        offload_optimizer=cpu_offload,
        offload_parameters=cpu_offload,
        config=deepspeed_config
    )