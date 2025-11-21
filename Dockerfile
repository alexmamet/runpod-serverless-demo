FROM runpod/base:0.6.3-cuda11.8.0

COPY pyproject.toml .

RUN pip install --no-cache-dir uv
RUN uv pip install -r pyproject.toml --system

COPY hello_world.py handler.py

CMD python -u /handler.py
