FROM python:3.11-slim

# Install system dependencies, Node.js 22.16.x, Docker, and development tools
RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    ca-certificates \
    gnupg \
    lsb-release \
    git \
    vim \
    tree \
    jq \
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

# Install uv (uvx is included)
RUN pip install --no-cache-dir uv

# Create workspace directory for user-mounted content
RUN mkdir -p /workspace

# Set working directory
WORKDIR /app

# Copy all necessary files for the build
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./
COPY src/ ./src/

# Install dependencies and build smolval
RUN uv sync --frozen --no-dev && \
    uv build && \
    uv pip install --system dist/*.whl

# Install Claude Code CLI and common MCP servers globally for faster startup
RUN npm install -g \
    @anthropic-ai/claude-code \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-memory

# Configure Claude CLI environment and verify installation
ENV CLAUDE_CONFIG_DIR=/app/.claude
RUN mkdir -p $CLAUDE_CONFIG_DIR \
    && claude --version

# Note: Users should mount their workspace with -v $(pwd):/workspace
# Example prompts and configs are available in the source repository

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user and fix permissions
RUN useradd --create-home --shell /bin/bash smolval && \
    chown -R smolval:smolval /app

# Pre-pull MCP Docker images for faster startup (optional, requires Docker socket)
# RUN docker pull mcp/sqlite || true

# Add user to docker group for Docker-in-Docker support
RUN usermod -a -G docker smolval 2>/dev/null || true

# Switch to non-root user for security
USER smolval

# Usage:
# Basic: docker run --rm -v $(pwd):/workspace -e ANTHROPIC_API_KEY smolval eval /workspace/prompts/test.txt
# With Docker MCP servers: docker run --rm -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock -e ANTHROPIC_API_KEY smolval eval /workspace/prompts/test.txt
# MCP configuration: Use .mcp.json files in your workspace, no config directory needed

# Set entrypoint to use the installed smolval command
ENTRYPOINT ["/app/.venv/bin/smolval"]
