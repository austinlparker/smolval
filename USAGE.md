# Quick Usage Guide

## Setup

1. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

2. **Build the container**:
   ```bash
   docker-compose build
   ```

## Usage Patterns

### Quick Evaluation
```bash
# Run a single evaluation
docker-compose run --rm eval eval prompts/simple_test.txt

# With custom format
docker-compose run --rm eval eval prompts/test.txt --format json

# With custom MCP config
docker-compose run --rm eval eval prompts/test.txt --mcp-config .mcp.json
```

### Interactive Development
```bash
# Start development container
docker-compose up dev -d

# Get a shell in the running container
docker-compose exec dev bash

# When done
docker-compose down
```

### One-off Shell
```bash
# Get an immediate interactive shell
docker-compose run --rm shell
```

## MCP Configuration

Create a `.mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {}
    }
  }
}
```

## File Structure

Your project should look like:
```
your-project/
├── .env                 # Your API key
├── .mcp.json           # MCP server config (optional)
├── prompts/            # Your prompt files
│   └── test.txt
└── results/            # Generated results (auto-created)
```

The container mounts your current directory as `/workspace`, so all paths should be relative to your project root.