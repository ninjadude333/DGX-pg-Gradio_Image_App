#!/usr/bin/env python3
import os
import sys
import traceback

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from huggingface_hub import constants

# Optional: make sure we are NOT in offline mode here
os.environ.pop("HF_HUB_OFFLINE", None)

# Same model config as in the app
MODEL_CONFIGS = {
	"Juggernaut XL (stablediffusionapi)": {
	    "repo_id": "stablediffusionapi/juggernautxl",
	    "kind": "sdxl",
	    "is_turbo": False,
	},
}


def print_header():
    print("=" * 80)
    print("SDXL model pre-download script")
    print("=" * 80)
    print()
    print("Hugging Face cache dir: {}".format(constants.HF_HUB_CACHE))
    print("Note: in your docker run, you mount:")
    print("  -v /root/.cache/huggingface:/root/.cache/huggingface")
    print()
    print("This script will try to download all models into that cache.")
    print("Make sure you have internet access and, if needed, HF_TOKEN configured.")
    print()


def try_download_txt2img(model_id: str):
    """
    Try to download the base SDXL pipeline.
    First attempt: use_safetensors=True
    If that fails due to missing safetensors, retry without that flag.
    """
    print("  [txt2img] Attempting with safetensors=True...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=None,  # dtype does not matter for download
            use_safetensors=True,
        )
        del pipe
        print("  [txt2img] OK (safetensors)")
        return True
    except Exception as e:
        msg = str(e)
        print("  [txt2img] Failed with safetensors=True: {}".format(msg))

        # If error mentions safetensors missing, retry without that option
        if "safetensors" in msg.lower():
            print("  [txt2img] Retrying without use_safetensors...")
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=None,
                )
                del pipe
                print("  [txt2img] OK (without safetensors)")
                return True
            except Exception as e2:
                print("  [txt2img] Still failed without safetensors: {}".format(e2))
                return False
        else:
            return False


def try_download_img2img(model_id: str):
    """
    Try to download the SDXL Img2Img pipeline.
    Same safetensors retry logic as txt2img.
    """
    print("  [img2img] Attempting with safetensors=True...")
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=None,
            use_safetensors=True,
        )
        del pipe
        print("  [img2img] OK (safetensors)")
        return True
    except Exception as e:
        msg = str(e)
        print("  [img2img] Failed with safetensors=True: {}".format(msg))

        if "safetensors" in msg.lower():
            print("  [img2img] Retrying without use_safetensors...")
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=None,
                )
                del pipe
                print("  [img2img] OK (without safetensors)")
                return True
            except Exception as e2:
                print("  [img2img] Still failed without safetensors: {}".format(e2))
                return False
        else:
            return False


def main():
    print_header()

    success_count = 0
    fail_count = 0

    for name, cfg in MODEL_CONFIGS.items():
        model_id = cfg["repo_id"]
        print("-" * 80)
        print("Model: {}".format(name))
        print("Repo : {}".format(model_id))
        print()

        # Some models may not exist anymore (404).
        # We handle errors gracefully and move on.
        try:
            ok_txt2img = try_download_txt2img(model_id)
            ok_img2img = try_download_img2img(model_id)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(1)
        except Exception:
            print("  Unexpected error:")
            traceback.print_exc()
            ok_txt2img = False
            ok_img2img = False

        if ok_txt2img or ok_img2img:
            print("  => Finished: at least one pipeline downloaded for {}".format(name))
            success_count += 1
        else:
            print("  => FAILED to download pipelines for {}".format(name))
            fail_count += 1

        print()

    print("=" * 80)
    print("Done.")
    print("Models with at least one successful pipeline: {}".format(success_count))
    print("Models that failed completely: {}".format(fail_count))
    print("HF cache dir: {}".format(constants.HF_HUB_CACHE))
    print("=" * 80)


if __name__ == "__main__":
    main()

