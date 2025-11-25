import base64

import runpod
import torch
import time
from PIL import Image
import io
from diffusers import QwenImageEditPipeline
from huggingface_hub import hf_hub_download
import sys
from loguru import logger
import httpx

logger.debug(sys.executable)

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipeline = pipeline.to(torch.bfloat16)
pipeline = pipeline.to("cuda")
logger.debug("Qwen downloaded")

lora_path_step = hf_hub_download(
    repo_id="lightx2v/Qwen-Image-Lightning",
    filename="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
)
logger.debug("Lora 1 downloaded")
lora_path_nsfw = hf_hub_download(
    repo_id="mlgethoney/qwen-lora-nsfw",
    filename="qwen_big_run_v1_3200+14600_edit_plus-step00014000.safetensors",
)
logger.debug("Lora 2 downloaded")
pipeline.load_lora_weights(lora_path_step, adapter_name="steps")
pipeline.load_lora_weights(lora_path_nsfw, adapter_name="nsfw")
pipeline.set_adapters(["steps", "nsfw"], adapter_weights=[1.0, 1.0])
logger.debug("Lora downloaded")

pipeline.fuse_lora()
logger.debug("Lora Fused")


def handler(job):
    """
    Handler that processes image editing requests.
    Accepts either image URL or base64 encoded image.
    The job parameter contains the input data in job["input"]
    """
    job_input = job["input"]
    logger.info(f"Recv {str(job_input)[:250]}...")

    # Check if input is URL or base64
    image_input = job_input.get("image") or job_input.get("image_url")

    if image_input.startswith(("http://", "https://")):
        # Download image from URL using httpx
        logger.info(f"Downloading image from URL: {image_input}")
        with httpx.Client(timeout=30.0) as client:
            response = client.get(image_input)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
    else:
        # Decode base64 image
        logger.info("Processing base64 encoded image")
        image_bytes = base64.b64decode(image_input)
        image = Image.open(io.BytesIO(image_bytes))

    t0 = time.time()
    output = pipeline(
        image=image,
        prompt=job_input["prompt"],
        num_inference_steps=job_input.get("num_inference_steps", 8),
        generator=torch.manual_seed(10**4),
    )
    logger.info(f"Image generated. Time taken: {time.time() - t0}")
    output_image = output.images[0]
    output_image_base64 = base64.b64encode(output_image.tobytes()).decode("ascii")
    return {"image_base64": output_image_base64}


runpod.serverless.start({"handler": handler})
