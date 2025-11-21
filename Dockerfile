FROM runpod/base:0.6.3-cuda11.8.0

COPY pyproject.toml .


RUN pip install --no-cache-dir uv
RUN uv pip install 'huggingface_hub[cli,torch]' --system

#base model
RUN hf download Qwen/Qwen-Image-Edit

#lora models
RUN hf download mlgethoney/qwen-lora-nsfw qwen_big_run_v1_3200+14600_edit_plus-step00014000.safetensors
RUN hf download lightx2v/Qwen-Image-Lightning Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors


RUN uv pip install -r pyproject.toml --system

COPY handler.py hello_world.py

CMD python -u /hello_world.py
