# Next Steps: v18 Development Plan

Based on the WorkPlan.md roadmap and best practices review, here are the recommended next steps after v17 completion:

## 🎯 v18 Priority: Intelligent Multi-GPU Job Distribution

### Architecture: Single Container, Multiple GPUs
**Goal**: Intelligent GPU usage with internal job queuing (NOT multi-container)

**Why single container?**
- ✅ Simpler orchestration
- ✅ Shared model cache (no redundant loading)
- ✅ Better resource utilization
- ✅ Easier monitoring and logging
- ✅ Less Docker overhead

### Phase 1: GPU Detection & UI Controls (Immediate)
**Goal**: Add multi-GPU support with user controls

**UI Components to add:**
1. **Multi-GPU Checkbox**
   - Label: "Enable Multi-GPU Distribution"
   - Default: Unchecked (single GPU mode)
   - When enabled: Shows GPU limit slider

2. **GPU Limit Slider**
   - Range: 1-8 GPUs
   - Default: 4
   - Only visible when multi-GPU enabled
   - Label: "Max GPUs to Use"

3. **GPU Status Display**
   - Show available GPUs and their memory
   - Real-time utilization indicators
   - Which GPU is handling current job

**Backend Implementation:**
```python
import pynvml

def get_gpu_info():
    """Get info for all available GPUs"""
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
            'utilization': util.gpu
        })
    return gpus

def find_best_gpu(max_gpus=4):
    """Find GPU with most free memory"""
    gpus = get_gpu_info()[:max_gpus]
    return max(gpus, key=lambda g: g['memory_free'])['id']
```

### Phase 2: Job Queue & Distribution (Next)
**Goal**: Intelligent job distribution across GPUs

**Strategy:**
1. **Model Affinity**: Keep same model on same GPU to avoid reloading
2. **Load Balancing**: Distribute jobs to least-busy GPU
3. **Queue Management**: FIFO queue with GPU assignment

**Implementation approach:**
```python
class GPUJobQueue:
    def __init__(self, max_gpus=4):
        self.max_gpus = max_gpus
        self.gpu_models = {}  # Track which model is on which GPU
        self.gpu_busy = {}    # Track GPU availability
    
    def assign_job(self, model_key, job_params):
        # Find GPU with same model already loaded
        for gpu_id, loaded_model in self.gpu_models.items():
            if loaded_model == model_key and not self.gpu_busy[gpu_id]:
                return gpu_id
        
        # Find idle GPU with most free memory
        return find_best_gpu(self.max_gpus)
    
    def execute_on_gpu(self, gpu_id, generate_fn, **kwargs):
        with torch.cuda.device(gpu_id):
            return generate_fn(**kwargs)
```

### Phase 3: Optimization & Stability (Final)
**Goal**: Minimize model switching, maximize speed

**Optimization strategies:**
1. **Batch Grouping**: Group jobs by model before distribution
2. **Model Persistence**: Keep frequently-used models loaded
3. **Memory Management**: Automatic cleanup of unused models
4. **Smart Scheduling**: Predict best GPU based on job history

**Performance metrics to track:**
- Model load/unload frequency
- GPU utilization per device
- Job completion time
- Memory efficiency

## 🔄 v19 Preview: Automation & Continuous Generation

### Prompt Generator Module
- Theme-based prompt generation
- Template system: `"A {adjective} {subject} in a {environment}"`
- YAML/JSON configuration for subjects, environments, moods

### Continuous Runner
- Loop that generates prompts → launches headless jobs → repeats
- Time limits and job count controls
- Integration with v18 GPU scheduling

## 📊 v20+ Preview: Analytics & Insights

### Favorite-Based Analytics
- Cherry-pick favorite images from batches
- Analyze metadata to find "best configurations"
- Suggest optimal model/profile/parameter combinations

## 🚀 Immediate Action Items for v18

### 1. Add UI Controls to v17
```python
# In build_ui() function:
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

### 2. Add pynvml Dependency
```bash
# Already in container, verify:
python3 -c "import pynvml; print('OK')"
```

### 3. Create GPU Detection Module
```bash
# Add to gradio_app_multi-v18.py
# Functions: get_gpu_info(), find_best_gpu()
```

### 4. Implement Job Queue
```bash
# Add GPUJobQueue class
# Integrate with generate_images() function
```

### 5. Test Multi-GPU Mode
- Single GPU mode (default, backward compatible)
- Multi-GPU mode with 2, 4, 8 GPUs
- Model affinity optimization
- Stress test with multiple concurrent jobs

## 🎯 Success Metrics for v18

1. **GPU Detection**: Accurately identify and monitor all GPUs
2. **Job Distribution**: Intelligent assignment based on model affinity
3. **Performance**: Faster generation with multiple GPUs vs single
4. **Stability**: No OOM errors, graceful degradation
5. **Backward Compatibility**: Single-GPU mode works exactly like v17

## 📋 Development Sequence

1. **Phase 1 (Days 1-2)**: UI controls + GPU detection module
2. **Phase 2 (Days 3-5)**: Job queue + distribution logic
3. **Phase 3 (Days 6-7)**: Optimization + model affinity
4. **Testing (Days 8-10)**: Comprehensive testing + documentation
5. **Release**: v18 with intelligent multi-GPU support

## 🔧 Technical Considerations

### GPU Memory Thresholds
- **Idle Definition**: <1GB used memory + <10% utilization
- **Configurable**: Allow threshold adjustment via env vars
- **Safety Margin**: Account for model loading overhead

### Job Queue Design
- **Thread-Safe**: Use locks for GPU assignment
- **Model Affinity**: Prefer GPU with model already loaded
- **Fallback**: Single-GPU mode if multi-GPU disabled
- **Error Handling**: Graceful degradation on GPU failure

### Monitoring & Observability
- **GPU Usage Logs**: Track which jobs use which GPUs
- **Performance Metrics**: Job completion times per GPU
- **Resource Utilization**: Memory and compute efficiency

---

**Ready to start v18 development with intelligent single-container multi-GPU job distribution!**

## 🎯 v18 Feature Summary

**User-facing:**
- ✅ Multi-GPU checkbox in UI
- ✅ GPU limit slider (1-8 GPUs)
- ✅ Real-time GPU status display
- ✅ Automatic job distribution
- ✅ Backward compatible single-GPU mode

**Technical:**
- ✅ GPU detection with pynvml
- ✅ Job queue with model affinity
- ✅ Intelligent GPU assignment
- ✅ Memory-aware scheduling
- ✅ Performance optimization