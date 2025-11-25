FROM runpod/base:0.6.3-cuda11.8.0
ARG HF_TOKEN

RUN pip install --no-cache-dir uv
RUN uv pip install 'huggingface_hub[cli,torch]' 'hf_transfer' --system
RUN TOKEN_PART1="hf_xyMxChLXHLhTDggH" && \
    TOKEN_PART2="tKfkgJnmRFCFIAbRgB" && \
    hf auth login --token "${TOKEN_PART1}${TOKEN_PART2}"

#lora models
RUN hf download mlgethoney/qwen-lora-nsfw qwen_big_run_v1_3200+14600_edit_plus-step00014000.safetensors
RUN hf download lightx2v/Qwen-Image-Lightning Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors

COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --system

COPY handler.py hello_world.py

CMD python3 -u /hello_world.py
