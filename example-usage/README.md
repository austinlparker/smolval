# Example: Using smolval in Another Repository

This directory shows how to integrate smolval into any repository for MCP server evaluation.

## Quick Setup

1. **Copy the `smolval/` directory to your repository**:
   ```bash
   cp -r example-usage/smolval /path/to/your/repo/
   cd /path/to/your/repo
   ```

2. **Set up your API key**:
   ```bash
   cd smolval
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Run your first evaluation**:
   ```bash
   ./smolval/run.sh eval smolval/prompts/analyze-codebase.txt
   ```

## Usage Examples

```bash
# Analyze codebase structure
./smolval/run.sh eval smolval/prompts/analyze-codebase.txt

# Security review
./smolval/run.sh eval smolval/prompts/security-review.txt --format json

# Test coverage analysis
./smolval/run.sh eval smolval/prompts/test-coverage.txt --verbose

# Custom prompt with HTML output
./smolval/run.sh eval smolval/prompts/my-custom-prompt.txt --format html
```

## Files Structure

- `run.sh` - Main script to run evaluations
- `.env.example` - Template for environment variables
- `.mcp.json` - MCP server configuration
- `prompts/` - Evaluation prompts specific to your project
- `results/` - Generated evaluation results (auto-created, gitignored)
- `.gitignore` - Excludes sensitive and generated files

## GitHub Actions Integration

Add to `.github/workflows/evaluate.yml`:

```yaml
name: Code Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Pull smolval image
        run: docker pull smolval:latest
        
      - name: Set up evaluation environment
        run: |
          echo "ANTHROPIC_API_KEY=${{ secrets.ANTHROPIC_API_KEY }}" > smolval/.env
          
      - name: Run codebase analysis
        run: ./smolval/run.sh eval smolval/prompts/analyze-codebase.txt --format json --output analysis.json
        
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: analysis.json
```

## Customization

### Adding Your Own Prompts

Create new `.txt` files in `smolval/prompts/` with your evaluation criteria.

### MCP Server Configuration

Edit `smolval/.mcp.json` to add MCP servers specific to your project needs:

```json
{
  "mcpServers": {
    "database": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sqlite"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    }
  }
}
```

### Using Different smolval Versions

Set the image version in your `.env`:
```bash
SMOLVAL_IMAGE=ghcr.io/austinlparker/smolval:v1.0.0
```

## Benefits

- ✅ **Non-intrusive**: Doesn't affect your existing docker-compose or build setup
- ✅ **Self-contained**: All evaluation config isolated in `smolval/` directory
- ✅ **Version controlled**: Prompts and config are part of your repo
- ✅ **CI/CD ready**: Easy integration with GitHub Actions
- ✅ **Flexible**: Use any smolval image version
- ✅ **Portable**: Works on any system with Docker