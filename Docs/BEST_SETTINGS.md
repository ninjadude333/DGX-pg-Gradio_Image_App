# Model-Defined Constraints (Fixed)
Hardcoded in model architecture:
Resolution divisibility: Must be divisible by 8 (VAE requirement)
Max resolution: Technically unlimited, but quality degrades beyond training resolution

### SDXL trained on: 1024×1024
### PixArt trained on: 1024×1024
### SD3 trained on: 1024×1024
## Steps range: No hard limit, but diminishing returns
### SDXL: 20-50 optimal
### PixArt: 24-36 optimal
### SD3: 28-40 optimal
## CFG range: Model-specific
### SDXL: 5-10 works
### PixArt: 3.5-5.5 (breaks above 6.0)
### SD3: 5-9 works

## CFG (Classifier-Free Guidance) Explained
CFG Scale = How strictly the model follows your prompt
What It Does
## Low CFG (1-3): Model ignores prompt, generates "whatever it wants"
## Medium CFG (5-8): Balanced - follows prompt but stays creative
## High CFG (10-20): Strictly follows prompt, can become oversaturated/distorted
Unconditional: What model generates with NO prompt
Conditional: What model generates WITH your prompt

## CFG scale: Multiplier for the difference
## Model-Specific Ranges
### SDXL (5-10 optimal):
### CFG 5-7: Natural, balanced
### CFG 8-10: Strong prompt adherence
### CFG >12: Oversaturated, artifacts
### PixArt (3.5-5.5 optimal, NEVER >6.0):
### CFG 3.5-4.5: Soft, artistic
### CFG 5.0-5.5: Balanced
### CFG >6.0: BREAKS - severe artifacts, corruption
### SD3 (5-9 optimal):
### CFG 5-7: Natural
### CFG 7-9: Strong adherence
### CFG >10: Oversaturated

# Practical Guide
## Start at 7.0 for SDXL/SD3
## Start at 4.5 for PixArt

Increase if output ignores prompt
Decrease if output looks oversaturated/artificial


# Summary
## SDXL
STEPS 20-50
RES 1024×1024
CFG 5-10

## PixArt
STEPS 24-36
RES 1024×1024
CFG 3.5-5.5: Balanced

## SD3
STEPS 28-40
RES 1024×1024
CFG 5-9: Natural