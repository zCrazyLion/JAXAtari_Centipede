########################
# 1. builder stage
########################
FROM python:3.10-slim AS builder

# system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# copy the minimal files that change dependencies first (better caching)
COPY pyproject.toml README.md ./

# install build front-end
RUN pip install --upgrade pip build

# copy the rest of your code and tests
COPY . .

# build a wheel (and an sdist if you like)
RUN python -m build --wheel --outdir /tmp/wheels

########################
# 2. runtime / test stage
########################
FROM python:3.10-slim

# if your tests need git, SDL libs, etc. add them here
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# install the wheel and the extra CI requirements
COPY --from=builder /tmp/wheels /tmp/wheels
RUN set -e ; \
    pip install --no-cache-dir -U pip ; \
    WHEEL=$(ls /tmp/wheels/*.whl | head -n 1) ; \
    pip install --no-cache-dir "$WHEEL[gh_ci]"

RUN mkdir -p /src
WORKDIR /src

# run tests by default (exec form avoids an extra shell and quoting issues)
ENTRYPOINT ["pytest"]
