FROM python:3.11-slim

# Install system dependencies, Node.js 22.16.x, and Docker
RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    ca-certificates \
    gnupg \
    lsb-release \
    && NODE_VERSION=22.16.0 \
    && ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then NODE_ARCH="x64"; elif [ "$ARCH" = "arm64" ]; then NODE_ARCH="arm64"; else echo "Unsupported architecture: $ARCH" && exit 1; fi \
    && curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz | tar -xJ -C /usr/local --strip-components=1 \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install common MCP servers globally for faster startup
RUN npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-memory

# Set working directory
WORKDIR /app

# Copy all necessary files for the build
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy additional runtime files
COPY config/ ./config/
COPY prompts/ ./prompts/

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user and fix permissions
RUN useradd --create-home --shell /bin/bash smolval && \
    chown -R smolval:smolval /app

# Pre-pull MCP Docker images for faster startup (optional, requires Docker socket)
# RUN docker pull mcp/sqlite || true

# Note: For Docker-in-Docker to work properly, the container may need to run as root
# or with the correct user/group IDs. See CLAUDE.md for usage instructions.
USER smolval

# Set entrypoint
ENTRYPOINT ["uv", "run", "smolval"]