#!/bin/bash
set -e

# smolval runner script
# This script runs smolval evaluations from within any repository

# Change to smolval directory to ensure proper context
cd "$(dirname "$0")"

# Default to latest if no version specified
SMOLVAL_IMAGE="${SMOLVAL_IMAGE:-ghcr.io/austinlparker/smolval:latest}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found in smolval directory"
    echo "Please create .env with your ANTHROPIC_API_KEY"
    echo "Example: echo 'ANTHROPIC_API_KEY=your-key-here' > .env"
    exit 1
fi

# Check if API key is set
if ! grep -q "ANTHROPIC_API_KEY=" .env || grep -q "ANTHROPIC_API_KEY=$" .env; then
    echo "❌ Error: ANTHROPIC_API_KEY not set in .env file"
    echo "Please add your API key to .env: ANTHROPIC_API_KEY=your-key-here"
    exit 1
fi

echo "🚀 Running smolval with image: $SMOLVAL_IMAGE"

# Run smolval with proper volume mounts
# - Mount parent directory as /workspace (the actual repo)
# - Mount our .env file
# - Mount our .mcp.json if it exists
# - Enable Docker-in-Docker for MCP servers
# Note: The container entrypoint is already "uv run smolval", so we just pass arguments
# Check if .mcp.json exists and is valid JSON
USE_MCP_CONFIG=false
if [ -f .mcp.json ]; then
    # Check if file is valid JSON and not empty
    if [ -s .mcp.json ] && python3 -m json.tool .mcp.json >/dev/null 2>&1; then
        USE_MCP_CONFIG=true
        echo "📋 Using MCP configuration: .mcp.json"
    else
        echo "⚠️  Warning: .mcp.json exists but is empty or invalid JSON - skipping MCP config"
    fi
else
    echo "ℹ️  No .mcp.json found - using Claude Code's built-in tools only"
fi

# Parse arguments to adjust paths for the new working directory
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --output=* ]]; then
        # Extract the output path and make it relative to workspace root
        output_path="${arg#--output=}"
        # If path doesn't start with smolval/, prefix it
        if [[ "$output_path" != smolval/* ]]; then
            ARGS+=("--output=../${output_path}")
        else
            # Remove smolval/ prefix since we're now in /workspace/smolval
            output_path="${output_path#smolval/}"
            ARGS+=("--output=${output_path}")
        fi
    elif [[ "$arg" == "--output" ]]; then
        # Flag form - next argument will be the path
        ARGS+=("$arg")
        next_is_output=true
    elif [[ "$next_is_output" == true ]]; then
        # This is the output path argument
        if [[ "$arg" != smolval/* ]]; then
            ARGS+=("../${arg}")
        else
            # Remove smolval/ prefix since we're now in /workspace/smolval
            output_path="${arg#smolval/}"
            ARGS+=("${output_path}")
        fi
        next_is_output=false
    elif [[ "$arg" == eval ]]; then
        # Keep eval command as-is
        ARGS+=("$arg")
    elif [[ "$arg" == prompts/* || "$arg" == smolval/prompts/* ]]; then
        # Adjust prompt file paths relative to new working directory
        if [[ "$arg" == smolval/prompts/* ]]; then
            # Remove smolval/ prefix since we're now in /workspace/smolval
            prompt_path="${arg#smolval/}"
            ARGS+=("${prompt_path}")
        else
            # Path is already relative to workspace root, make it relative to smolval
            ARGS+=("../${arg}")
        fi
    else
        ARGS+=("$arg")
    fi
done

# Ensure smolval directory exists in workspace
mkdir -p ../smolval

# Run docker command with conditional MCP config
# Set working directory to /workspace/smolval so default results/ goes there
if [ "$USE_MCP_CONFIG" = true ]; then
    # Include MCP config and add the flag to arguments
    docker run --rm \
        -v "$(pwd)/..:/workspace" \
        -v "$(pwd)/.env:/workspace/smolval/.env" \
        -v "$(pwd)/.mcp.json:/workspace/smolval/.mcp.json" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        --env-file .env \
        -w /workspace/smolval \
        "$SMOLVAL_IMAGE" \
        "${ARGS[@]}" \
        --mcp-config .mcp.json
else
    # Run without MCP config
    docker run --rm \
        -v "$(pwd)/..:/workspace" \
        -v "$(pwd)/.env:/workspace/smolval/.env" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        --env-file .env \
        -w /workspace/smolval \
        "$SMOLVAL_IMAGE" \
        "${ARGS[@]}"
fi