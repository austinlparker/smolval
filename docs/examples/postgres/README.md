# PostgreSQL MCP Server Examples

This directory contains comprehensive examples for testing and comparing PostgreSQL MCP servers using smolval.

## Files Overview

- **[setup-postgres-testdata.sql](setup-postgres-testdata.sql)** - SQL script to create test database and sample data
- **[postgres-comparison.yaml](postgres-comparison.yaml)** - Configuration for comparing multiple PostgreSQL servers
- **[postgres-v1-only.yaml](postgres-v1-only.yaml)** - Configuration for testing original PostgreSQL server
- **[postgres-v2-only.yaml](postgres-v2-only.yaml)** - Configuration for testing PostgreSQL v2 server  
- **[postgres-comparison.txt](postgres-comparison.txt)** - Comprehensive prompt for testing PostgreSQL functionality

## Quick Start

### 1. Set Up Test Database

First, create a PostgreSQL database and load the test data:

```bash
# Create the test database
createdb smolval-test

# Load test data
psql smolval-test < docs/examples/postgres/setup-postgres-testdata.sql
```

### 2. Verify Docker Images

Ensure PostgreSQL MCP server Docker images are available:

```bash
# Check for existing images
docker images | grep postgres

# Pull the images if needed
docker pull mcp/postgres
docker pull mcp/postgresv2  # If testing v2 implementation
```

### 3. Run PostgreSQL Tests

#### Single Server Test
```bash
# Test PostgreSQL v1 server only
uv run smolval eval docs/examples/postgres/postgres-comparison.txt \
  -c docs/examples/postgres/postgres-v1-only.yaml

# Test PostgreSQL v2 server only  
uv run smolval eval docs/examples/postgres/postgres-comparison.txt \
  -c docs/examples/postgres/postgres-v2-only.yaml
```

#### Server Comparison
```bash
# Compare both PostgreSQL servers
uv run smolval compare \
  --baseline postgres \
  --test postgresv2 \
  docs/examples/postgres/postgres-comparison.txt \
  -c docs/examples/postgres/postgres-comparison.yaml \
  --format html
```

#### Batch Testing
```bash
# Create multiple test prompts and run as batch
mkdir postgres-tests
cp docs/examples/postgres/postgres-comparison.txt postgres-tests/
uv run smolval batch postgres-tests/ \
  -c docs/examples/postgres/postgres-comparison.yaml
```

## Test Database Schema

The setup script creates a comprehensive e-commerce database with:

### Tables
- **categories** (5 records) - Product categories
- **products** (13 records) - Products with JSON metadata  
- **customers** (5 records) - Customer information
- **orders** (5 records) - Order data with different statuses
- **order_items** (10 records) - Line items linking orders to products

### Views
- **customer_order_summary** - Aggregated customer statistics
- **product_sales_summary** - Product performance metrics

### Key Features
- **Realistic Data**: Includes edge cases, special characters, various price points
- **JSON Columns**: Tests JSON querying capabilities
- **Relationships**: Foreign keys between all related tables
- **Indexes**: Performance testing with proper indexing
- **Data Types**: Tests various PostgreSQL data types

## What Gets Tested

### Schema Discovery
- Table listing and structure inspection
- Column types, constraints, and relationships
- Index and view information
- Database metadata queries

### Query Capabilities
- Basic SELECT operations with filtering
- JOIN operations across multiple tables
- Aggregation functions (COUNT, SUM, AVG, etc.)
- GROUP BY and ORDER BY clauses
- Subqueries and Common Table Expressions (CTEs)
- Window functions
- JSON operations and querying

### Advanced Features
- Complex multi-table JOINs
- Performance with large result sets
- Error handling for invalid queries
- Connection pooling behavior
- Transaction handling

### Security Testing
- Write operation rejection (INSERT/UPDATE/DELETE)
- SQL injection prevention
- Permission boundary testing

## Configuration Details

### Environment Variables Required
```bash
# Database connection settings
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="smolval-test"
export POSTGRES_USER="your_username"
export POSTGRES_PASSWORD="your_password"

# API key for LLM
export ANTHROPIC_API_KEY="your_api_key"
```

### Docker Network Configuration
The PostgreSQL MCP servers run in Docker containers and need to connect to your local PostgreSQL instance:

- Uses `--net host` to access localhost PostgreSQL
- Requires PostgreSQL to accept connections from containers
- May need to configure `pg_hba.conf` for container access

## Performance Benchmarks

Expected performance characteristics:

- **Schema queries**: < 2 seconds
- **Simple SELECT**: < 1 second  
- **Complex JOINs**: < 5 seconds
- **Aggregations**: < 3 seconds
- **Large result sets**: < 10 seconds

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```
   Error: could not connect to server
   ```
   Solutions:
   - Verify PostgreSQL is running: `pg_isready`
   - Check database exists: `psql -l | grep smolval-test`
   - Test connection: `psql smolval-test -c "SELECT 1;"`

2. **Docker Container Cannot Reach Database**
   ```
   Error: connection refused
   ```
   Solutions:
   - Use `host.docker.internal` instead of `localhost` on macOS/Windows
   - Configure PostgreSQL to accept container connections
   - Check firewall settings

3. **Permission Denied**
   ```
   Error: permission denied for table
   ```
   Solutions:
   - Ensure test user has SELECT permissions
   - Check `pg_hba.conf` authentication settings
   - Verify user can connect: `psql -U testuser smolval-test`

4. **MCP Server Startup Issues**
   ```
   Error: failed to start MCP server
   ```
   Solutions:
   - Verify Docker is running
   - Check PostgreSQL MCP server image exists
   - Review environment variable configuration

### Debug Mode

Run with debug logging to troubleshoot issues:

```bash
uv run smolval --debug eval docs/examples/postgres/postgres-comparison.txt \
  -c docs/examples/postgres/postgres-v1-only.yaml
```

This shows:
- Database connection attempts
- SQL query execution
- MCP protocol messages
- Container startup logs

## Advanced Usage

### Custom Test Data

To test with your own data:

1. Modify `setup-postgres-testdata.sql` with your schema
2. Update the prompt file to test your specific use cases
3. Adjust performance expectations in success criteria

### Multi-Environment Testing

```bash
# Test against different PostgreSQL versions
POSTGRES_VERSION=13 docker run postgres:13 # Setup for v13
POSTGRES_VERSION=15 docker run postgres:15 # Setup for v15

# Compare performance across versions
uv run smolval compare --baseline postgres-v13 --test postgres-v15 \
  docs/examples/postgres/ --format html
```

### CI/CD Integration

```yaml
# .github/workflows/postgres-tests.yml
name: PostgreSQL MCP Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: smolval-test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: Load test data
      run: |
        psql postgresql://postgres:postgres@localhost:5432/smolval-test \
          < docs/examples/postgres/setup-postgres-testdata.sql
    
    - name: Run PostgreSQL tests
      run: |
        uv run smolval eval docs/examples/postgres/postgres-comparison.txt \
          -c docs/examples/postgres/postgres-v1-only.yaml
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        POSTGRES_HOST: localhost
        POSTGRES_USER: postgres
        POSTGRES_PASSWORD: postgres
        POSTGRES_DB: smolval-test
```

This PostgreSQL example provides comprehensive testing capabilities for database MCP servers with realistic data and robust evaluation criteria.