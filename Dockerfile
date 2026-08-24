FROM python:3.12.1-slim-bookworm

#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR  /app
ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./
# need no-install-project, predict.py and insurance_model.bin haven't been copied in yet
RUN uv sync --locked --no-install-project 

COPY "predict.py" "train.py" "insurance_model.bin" ./

EXPOSE 8080

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]