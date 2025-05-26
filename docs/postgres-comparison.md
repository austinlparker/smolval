# PostgreSQL MCP Server Comparison

This document provides guidance for comparing PostgreSQL MCP servers using smolval.

> **Note**: PostgreSQL examples and test data have been moved to [`docs/examples/postgres/`](examples/postgres/) for better organization.

## Setup

### 1. Prepare PostgreSQL Database
First, ensure you have PostgreSQL running locally on the default port (5432) and create the test database:

```bash
# Create the test database
createdb smolval-test

# Load test data
psql smolval-test < docs/examples/postgres/setup-postgres-testdata.sql
```

### 2. Verify Docker Images
Make sure both PostgreSQL MCP server images are available:

```bash
# Check for existing images
docker images | grep postgres

# Pull the images if needed
docker pull mcp/postgres
docker pull mcp/postgresv2  # Assuming this exists
```

## Running Comparisons

### Direct Server Comparison (Recommended)
Compare both servers against the same prompt:

```bash
uv run python -m smolval.cli compare --baseline postgres --test postgresv2 docs/examples/postgres/postgres-comparison.txt -c docs/examples/postgres/postgres-comparison.yaml --format markdown
```

### Individual Server Testing
Test each server separately:

```bash
# Test PostgreSQL v1
uv run python -m smolval.cli eval docs/examples/postgres/postgres-comparison.txt -c docs/examples/postgres/postgres-v1-only.yaml

# Test PostgreSQL v2
uv run python -m smolval.cli eval docs/examples/postgres/postgres-comparison.txt -c docs/examples/postgres/postgres-v2-only.yaml
```

### HTML Reports
For better visualization, use HTML format:

```bash
uv run python -m smolval.cli compare --baseline postgres --test postgresv2 docs/examples/postgres/postgres-comparison.txt -c docs/examples/postgres/postgres-comparison.yaml --format html
```

## Test Data Overview

The setup script creates a comprehensive e-commerce database with:

- **5 categories** (Electronics, Books, Clothing, Home & Garden, Sports)
- **13 products** with realistic data including JSON metadata
- **5 customers** with contact information
- **5 orders** with different statuses
- **10 order items** linking orders to products
- **Indexes** on key columns for performance testing
- **Views** for testing complex query capabilities
- **Edge cases** (special characters, extreme prices, free items)

## What Gets Tested

### Schema Discovery
- Table listing and structure
- Column types and constraints
- Indexes and relationships
- Database metadata

### Query Capabilities
- Basic SELECT operations
- Aggregations (COUNT, SUM, AVG, etc.)
- JOINs across multiple tables
- WHERE clauses and filtering
- Grouping and ordering
- Subqueries and CTEs
- Window functions
- JSON operations

### Error Handling
- Invalid table names
- Malformed SQL syntax
- Write operation attempts (should fail)
- Connection behavior

### Performance
- Query execution speed
- Large result set handling
- Complex query support

## Configuration Files

- `docs/examples/postgres/postgres-comparison.yaml` - Both servers for comparison
- `docs/examples/postgres/postgres-v1-only.yaml` - Only the original postgres server
- `docs/examples/postgres/postgres-v2-only.yaml` - Only the postgresv2 server

## Expected Results

Since these are read-only MCP servers:
- ✅ All SELECT queries should work
- ✅ Schema introspection should be available
- ❌ Write operations (INSERT/UPDATE/DELETE) should be rejected
- 📊 Performance and capability differences should be highlighted

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running: `pg_isready`
- Check database exists: `psql -l | grep smolval-test`
- Verify connection: `psql smolval-test -c "SELECT 1;"`

### Docker Issues
- Check containers can reach host: `docker run --rm postgres:latest psql -h host.docker.internal -U postgres -l`
- Ensure images exist: `docker images | grep postgres`

### Permission Issues
- Make sure PostgreSQL accepts connections from Docker containers
- Check `pg_hba.conf` if authentication fails

## Output

Comparison results will be generated in multiple formats:
- `results/[timestamp]/comparison_postgres-comparison.json` - Raw data
- `results/[timestamp]/comparison_postgres-comparison.md` - Markdown report  
- `results/[timestamp]/comparison_postgres-comparison.html` - Interactive HTML report

The HTML report provides the best visualization with collapsible sections, performance metrics, and side-by-side comparisons.

## See Also

For detailed PostgreSQL testing examples, configuration files, and setup instructions, see the comprehensive guide at [`docs/examples/postgres/`](examples/postgres/).