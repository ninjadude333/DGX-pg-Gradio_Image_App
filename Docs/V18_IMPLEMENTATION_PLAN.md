# v18 Implementation Plan: Intelligent Multi-GPU Job Distribution

## Architecture Decision: Single Container

**Approach:** Single container with internal job queue (NOT multi-container)

**Why?**
- ✅ Simpler orchestration
- ✅ Shared model cache (no redundant loading)
- ✅ Better resource utilization
- ✅ Easier monitoring and logging
- ✅ Less Docker overhead

## Phase 1: UI Controls & GPU Detection

### UI Components to Add

```python
# In build_ui() function, add after batch_size slider:

with gr.Row():
    enable_multi_gpu = gr.Checkbox(
        value=False,
        label="🚀 Enable Multi-GPU Distribution"
    )
    max_gpus = gr.Slider(
        minimum=1,
        maximum=8,
        value=4,
        step=1,
        label="Max GPUs to Use",
        visible=False
    )

# Toggle visibility
enable_multi_gpu.change(
    fn=lambda x: gr.update(visible=x),
    inputs=[enable_multi_gpu],
    outputs=[max_gpus]
)
```

### GPU Detection Functions

```python
import pynvml

def get_gpu_info():
    """Get info for all available GPUs"""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append({
                'id': i,
                'memory_free': mem_info.free,
                'memory_total': mem_info.total,
                'memory_used': mem_info.used,
                'utilization': util.gpu
            })
        return gpus
    except Exception as e:
        print(f"[WARNING] GPU detection failed: {e}")
        return []

def find_best_gpu(max_gpus=4):
    """Find GPU with most free memory"""
    gpus = get_gpu_info()[:max_gpus]
    if not gpus:
        return 0  # Fallback to default device
    return max(gpus, key=lambda g: g['memory_free'])['id']
```

## Phase 2: Job Queue & Distribution

### GPUJobQueue Class

```python
import threading
from typing import Dict, Optional

class GPUJobQueue:
    def __init__(self, max_gpus=4):
        self.max_gpus = max_gpus
        self.gpu_models: Dict[int, Optional[str]] = {}  # gpu_id -> model_key
        self.gpu_busy: Dict[int, bool] = {}  # gpu_id -> is_busy
        self.lock = threading.Lock()
        
        # Initialize GPU states
        for i in range(max_gpus):
            self.gpu_models[i] = None
            self.gpu_busy[i] = False
    
    def assign_job(self, model_key: str) -> int:
        """Assign job to best GPU based on model affinity"""
        with self.lock:
            # First: Find idle GPU with same model already loaded
            for gpu_id in range(self.max_gpus):
                if not self.gpu_busy[gpu_id] and self.gpu_models[gpu_id] == model_key:
                    self.gpu_busy[gpu_id] = True
                    return gpu_id
            
            # Second: Find any idle GPU
            for gpu_id in range(self.max_gpus):
                if not self.gpu_busy[gpu_id]:
                    self.gpu_busy[gpu_id] = True
                    self.gpu_models[gpu_id] = model_key
                    return gpu_id
            
            # Fallback: Use GPU with most free memory
            return find_best_gpu(self.max_gpus)
    
    def release_gpu(self, gpu_id: int):
        """Mark GPU as available"""
        with self.lock:
            self.gpu_busy[gpu_id] = False
    
    def execute_on_gpu(self, gpu_id: int, fn, *args, **kwargs):
        """Execute function on specific GPU"""
        try:
            with torch.cuda.device(gpu_id):
                result = fn(*args, **kwargs)
            return result
        finally:
            self.release_gpu(gpu_id)
```

## Phase 3: Integration with generate_images()

### Modified Generation Flow

```python
# Global job queue (initialized when multi-GPU enabled)
_gpu_queue: Optional[GPUJobQueue] = None

def generate_images(
    # ... existing parameters ...
    enable_multi_gpu: bool = False,
    max_gpus: int = 4,
):
    global _gpu_queue
    
    # Initialize queue if multi-GPU enabled
    if enable_multi_gpu and _gpu_queue is None:
        _gpu_queue = GPUJobQueue(max_gpus)
    
    # Single-GPU mode (backward compatible)
    if not enable_multi_gpu or _gpu_queue is None:
        # Existing single-GPU logic
        return _generate_single_gpu(...)
    
    # Multi-GPU mode
    return _generate_multi_gpu(...)

def _generate_multi_gpu(...):
    """Multi-GPU generation with job queue"""
    gpu_id = _gpu_queue.assign_job(model_key)
    print(f"[MULTI-GPU] Assigned to GPU {gpu_id}")
    
    try:
        result = _gpu_queue.execute_on_gpu(
            gpu_id,
            _generate_on_device,
            model_key=model_key,
            # ... other params
        )
        return result
    except Exception as e:
        print(f"[ERROR] GPU {gpu_id} failed: {e}")
        _gpu_queue.release_gpu(gpu_id)
        raise
```

## Implementation Checklist

### Phase 1 (Days 1-2)
- [ ] Add multi-GPU checkbox to UI
- [ ] Add GPU limit slider with conditional visibility
- [ ] Implement `get_gpu_info()` function
- [ ] Implement `find_best_gpu()` function
- [ ] Test GPU detection on DGX

### Phase 2 (Days 3-5)
- [ ] Implement `GPUJobQueue` class
- [ ] Add model affinity tracking
- [ ] Add GPU busy/idle state management
- [ ] Implement `assign_job()` method
- [ ] Implement `execute_on_gpu()` method

### Phase 3 (Days 6-7)
- [ ] Integrate queue with `generate_images()`
- [ ] Add backward compatibility (single-GPU mode)
- [ ] Test with 2, 4, 8 GPUs
- [ ] Add performance logging

### Testing (Days 8-10)
- [ ] Single-GPU mode (verify v17 compatibility)
- [ ] Multi-GPU mode with various GPU counts
- [ ] Model affinity optimization
- [ ] Stress test with concurrent jobs
- [ ] OOM handling and graceful degradation

## Success Metrics

1. **GPU Detection**: Accurately identify and monitor all GPUs
2. **Job Distribution**: Intelligent assignment based on model affinity
3. **Performance**: Faster generation with multiple GPUs vs single
4. **Stability**: No OOM errors, graceful degradation
5. **Backward Compatibility**: Single-GPU mode works exactly like v17

## Next Steps After v18

Once v18 is stable:
- v19: Automated prompt generator + continuous runner
- v20: Favorite-based analytics and insights

---

**Ready to implement v18 with single-container multi-GPU architecture!**
