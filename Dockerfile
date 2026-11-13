FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Pinned runtime dependencies for BaseAgent and Runner.
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps -r requirements.txt

COPY run_agent.py /app/run_agent.py
COPY harness /app/harness
# Framework invocation check.
RUN python -c "import asyncio; from harness.jiuwen import invoke_harness; assert asyncio.run(invoke_harness(lambda x: x, {'smoke': True})) == {'smoke': True}"

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
