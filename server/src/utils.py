import torch
from logger import logger
def check_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        logger.info(f"GPU Memory - Total: {total_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB, Cached: {cached_memory:.2f}GB")
        logger.info(f"Available: {total_memory - cached_memory:.2f}GB")
        return total_memory - cached_memory
    else:
        logger.info("CUDA not available")
        return 0

def print_model_vram(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 * 1024)

    logger.info(f"Model size: {total_size_mb:.2f} MB")
