# SDXL DGX Image Lab v17 - Release Notes

## 🎨 New Artist/Genre Profiles

### Tim Burton Style
- **Aesthetic**: Gothic, dark whimsical, striped patterns, pale skin, dark circles under eyes
- **Architecture**: Twisted, Burton-esque character design
- **Recommended Steps**: 30
- **Best For**: Character-focused scenes, gothic cityscapes, surreal environments

### Frank Frazetta Fantasy
- **Aesthetic**: Epic fantasy illustration, muscular heroes, dramatic poses, rich colors
- **Style**: Painterly brushstrokes, barbarian aesthetic
- **Scheduler**: DPM++ 2M (optimized)
- **Steps**: 35
- **Best For**: Fantasy warriors, epic monsters, heavy metal cover art

### Ralph Bakshi Animation
- **Aesthetic**: 1970s rotoscoped animation, gritty urban fantasy, detailed backgrounds
- **Style**: Adult animation, atmospheric
- **Steps**: 28
- **Best For**: Urban fantasy, surreal animation shots, stylized storytelling

### H.R. Giger Biomechanical
- **Aesthetic**: Alien, biomechanical, organic machinery, xenomorph aesthetic
- **Style**: Airbrushed, monochromatic, nightmarish beauty
- **Scheduler**: Euler (optimized)
- **Steps**: 40
- **Best For**: Sci-fi horror, alien structures, biomechanical beings

### Dark Fantasy / Grimdark
- **Aesthetic**: Ominous atmosphere, muted colors, gothic horror, medieval darkness
- **Style**: Foreboding mood, dramatic shadows
- **Scheduler**: DPM++ 2M (optimized)
- **Steps**: 32
- **Best For**: Dark fantasy scenes, grimdark atmospheres

## 🎨 Extended v16 Profiles (Also Included)

### Creative & Artistic Styles
- **Watercolor** - Soft watercolor painting with flowing colors
- **Hyper-Realistic Portrait** - Ultra-detailed portraits with perfect lighting
- **Pixar-ish Soft CG** - Family-friendly 3D rendering style
- **Children's Book Illustration** - Whimsical, colorful storybook art

### Technical & Gaming Styles
- **ISOTOPIA Sci-Fi Blueprint** - Technical schematics and futuristic designs
- **Pixel Art / Isometric Game** - Retro 16-bit gaming aesthetics
- **Low-Poly 3D / PS1** - Nostalgic 90s PlayStation graphics
- **Product Render / Industrial** - Clean commercial photography
- **Isometric Tech Diagram** - Engineering drawings and blueprints

### Pop Culture & Retro Styles
- **Retro Comic / Halftone** - Vintage comic book with halftone dots
- **Vaporwave / Synthwave** - 80s neon aesthetic and cyberpunk vibes
- **Ink & Screentone Manga** - Japanese comic art with detailed linework
- **Analog Horror / VHS** - Found footage horror with grainy distortion

### Professional Styles
- **Architectural Visualization** - Clean professional renders with realistic materials

## 🤖 Headless Mode for Automation

### Environment Variable Control
```bash
-e HEADLESS=true \
-e PROMPT="a cyberpunk city at night" \
-e MODEL="CyberRealistic XL 5.8" \
-e STYLE_PROFILE="Tim Burton Style" \
-e BATCH_SIZE=6 \
-e STEPS=30 \
-e CFG_SCALE=7.5 \
-e WIDTH=1024 \
-e HEIGHT=576 \
-e SEED=12345
```

### Batch Processing Options
- `RUN_ALL_PROFILES=true` - Run all profiles for selected model
- `RUN_ALL_MODELS=true` - Run all models × all profiles (comprehensive sweep)
- Automatic priority handling if both are set

### Perfect For
- Overnight batch jobs
- Automated workflows
- Continuous generation pipelines
- Scheduled tasks

## 🛡️ Safety & UX Improvements

### Mutually Exclusive Checkboxes
- "Run ALL profiles" and "Run ALL models × profiles" cannot both be active
- Automatic UI logic prevents conflicts
- Clear visual feedback
- Prevents accidental massive generation jobs

### Per-Instance Logging
- Uses `INSTANCE_ID` environment variable or hostname
- Logs to `jobs_{instance_id}.log` instead of shared `jobs.log`
- Perfect for multiple containers running simultaneously
- Prevents log conflicts in multi-container deployments

## 🔧 Technical Improvements

### Single-GPU Optimization
- Removed multi-GPU complexity for better stability
- Enhanced memory management with proper cleanup
- xformers memory optimization when available
- Lazy-loaded Img2Img pipeline to save VRAM

### Enhanced Profile System
- Each profile includes custom prompt/negative suffixes
- Optimal scheduler recommendations per profile
- Suggested step counts for best results
- Artist-specific styling and aesthetic guidance

### Thread-Safe Architecture
- Global locks for model loading safety
- Proper state management across UI sessions
- Consistent behavior in multi-container environments

## 📊 Enhanced Logging & Metadata

### Comprehensive Job Tracking
- Instance-specific log files prevent conflicts
- Detailed JSON logging with all parameters
- Organized output directories with timestamps
- Full reproducibility data including:
  - Original and styled prompts
  - Model, scheduler, and profile used
  - All generation parameters and seeds
  - File paths and run directories
  - Instance identification

### File Organization
- Timestamped run directories
- Model-specific subdirectories for multi-model runs
- Consistent naming convention: `{timestamp}_{profile}_{prompt}_{seed}_{index}.png`

## 🚀 Usage Examples

### UI Mode with New Profiles
1. Select "Tim Burton Style" profile
2. Enter prompt: "a gothic cathedral"
3. Notice automatic negative prompt injection
4. Generate with optimized 30 steps

### Headless Batch Job
```bash
sudo docker run --name image_gen_v17_headless \
  --gpus all --runtime=nvidia \
  -e CUDA_VISIBLE_DEVICES=4 \
  -e HEADLESS=true \
  -e PROMPT="a biomechanical alien landscape" \
  -e STYLE_PROFILE="H.R. Giger Biomechanical" \
  -e BATCH_SIZE=8 \
  -e INSTANCE_ID=giger_job_1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)":/app -w /app \
  gradio_app_generic:dude \
  python3 gradio_app_multi-v17.py
```

### Multi-Profile Sweep
```bash
-e RUN_ALL_PROFILES=true \
-e MODEL="Frank Frazetta Fantasy" \
-e PROMPT="a warrior in an epic battle"
```

## 🔄 Migration from v16

### Breaking Changes
- None - v17 is fully backward compatible

### New Features Available
- 5 new artist/genre profiles in dropdown
- All 14 v16 extended profiles preserved
- Headless mode via environment variables
- Enhanced safety with mutually exclusive options
- Per-instance logging (set `INSTANCE_ID` env var)

### Total Style Profiles Available
- **Core Profiles**: 10 (Photoreal, Cinematic, Anime, etc.)
- **New v17 Artist Profiles**: 5 (Tim Burton, Frazetta, etc.)
- **Extended v16 Profiles**: 14 (Watercolor, Pixel Art, etc.)
- **Total**: 29 style profiles for maximum creative control

### Recommended Updates
- Add `INSTANCE_ID` to your docker run commands
- Try the new artist profiles for enhanced creative control
- Use headless mode for automation workflows
- Explore the extended v16 profiles for specialized aesthetics

## 🎯 Next Steps (v18 Preview)

- GPU detection and scheduling
- Multi-GPU awareness without sharing
- External orchestrator scripts
- Enhanced job queuing system

---

**v17 delivers enhanced artistic control, automation capabilities, and production-ready safety features for the SDXL DGX Image Lab.**