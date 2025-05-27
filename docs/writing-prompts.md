# Writing Effective Evaluation Prompts

This guide covers how to write effective prompts for evaluating MCP servers using smolval.

## Prompt Structure

Evaluation prompts should be clear, actionable, and testable. They typically follow this structure:

```
[Context/Setup]
[Specific Tasks]
[Success Criteria]
```

## Basic Example

```
Test the basic functionality of the MCP servers by performing these tasks:

1. List the files in the current directory to understand the project structure
2. If you find any interesting files (like README, documentation, or example files), read their contents
3. Try to fetch content from a simple webpage (like https://example.com) to test the web fetching capability
4. Summarize what you learned about the project and the capabilities of the MCP servers

This evaluation will help verify that the MCP servers are working correctly and can handle file operations, web requests, and provide useful responses.
```

## Prompt Design Principles

### 1. Be Specific and Actionable

**Good:**
```
Create a new file called "test.txt" with the content "Hello World" and then read it back to verify the content.
```

**Bad:**
```
Test file operations.
```

### 2. Include Clear Success Criteria

**Good:**
```
Search for all Python files in the project and count them. The evaluation succeeds if:
- At least 5 Python files are found
- The search includes subdirectories
- File paths are correctly reported
```

**Bad:**
```
Find Python files and see what happens.
```

### 3. Test Progressive Complexity

Structure prompts to test basic functionality first, then more complex operations:

```
1. List files in the current directory (basic file operations)
2. Read the contents of README.md (file content access)
3. Search for files containing "TODO" comments (text search)
4. Generate a summary report of findings (synthesis)
```

### 4. Handle Error Cases

Include scenarios that test error handling:

```
1. Try to read a file that doesn't exist
2. Attempt to access a restricted directory
3. Make a request to an invalid URL
4. Verify that appropriate error messages are returned
```

## Prompt Categories

### File System Testing

Test filesystem MCP servers with operations like:

```
File System Evaluation:

1. Navigate to the project root directory
2. Create a test directory called "evaluation_test"
3. Create files with different extensions (.txt, .json, .py)
4. List directory contents and verify all files are present
5. Read file contents and modify them
6. Clean up by removing the test directory

Success criteria:
- All file operations complete without errors
- File contents are accurately read and written
- Directory operations work correctly
```

### Web Content Fetching

Test web-fetching MCP servers:

```
Web Content Evaluation:

1. Fetch the homepage of a reliable website (e.g., https://httpbin.org/get)
2. Extract and summarize key information from the content
3. Try fetching a JSON endpoint (e.g., https://httpbin.org/json)
4. Test error handling with an invalid URL
5. Compare response times and content quality

Success criteria:
- Valid HTML/JSON content is retrieved
- Content is properly parsed and summarized
- Error cases are handled gracefully
- Response times are reasonable (< 10 seconds)
```

### Database Operations

Test database MCP servers:

```
Database Evaluation:

1. List all available tables in the database
2. Describe the schema of the "customers" table
3. Execute a simple SELECT query to get the first 5 customers
4. Perform a JOIN between customers and orders tables
5. Execute an aggregation query (count orders by status)
6. Test error handling with invalid SQL

Success criteria:
- Schema information is accurate and complete
- Queries return expected data types and formats
- Complex queries (JOINs, aggregations) work correctly
- SQL errors are handled appropriately
```

## Advanced Techniques

### Multi-Server Comparisons

When comparing multiple MCP servers, use prompts that highlight differences:

```
Comparative File Search Evaluation:

1. Search for all JavaScript files in the project
2. Count the total lines of code across all JS files
3. Find files that contain the word "async"
4. Generate a summary report with file counts and sizes

This prompt will help compare how different file system servers handle:
- Search functionality and patterns
- File content analysis
- Performance with large file sets
- Reporting and data presentation
```

### Performance Testing

Include timing and performance expectations:

```
Performance Evaluation:

1. List all files in a large directory (>100 files)
2. Measure and report the time taken for this operation
3. Read the contents of 10 different files sequentially
4. Perform the same file reads and measure total time

Expected performance:
- Directory listing: < 2 seconds
- File reads: < 1 second per file
- Total evaluation: < 30 seconds
```

### Integration Testing

Test how multiple MCP servers work together:

```
Integration Evaluation:

1. Use the filesystem server to read a list of URLs from a file
2. Use the web fetching server to retrieve content from each URL
3. Save the fetched content to new files using the filesystem server
4. Generate a summary report of all operations

This tests:
- Data flow between different MCP servers
- Error handling across server boundaries
- Coordination of multiple tool types
```

## Template Variables

You can use template variables in prompts for dynamic content:

```
Project Analysis for {{project_name}}:

1. Examine the project structure in {{project_path}}
2. Look for configuration files relevant to {{technology_stack}}
3. Analyze {{file_pattern}} files for common patterns
4. Generate insights specific to {{project_type}} projects

Variables:
- project_name: The name of the project being analyzed
- project_path: Root directory path
- technology_stack: Primary technologies used
- file_pattern: File types to focus on (e.g., "*.py", "*.js")
```

## Testing Your Prompts

Before using prompts in production:

1. **Test with mock data**: Verify prompts work with known file structures
2. **Check error scenarios**: Ensure prompts handle missing files, network issues
3. **Validate success criteria**: Confirm that success/failure is clearly determinable
4. **Time the evaluation**: Ensure prompts complete within reasonable timeframes
5. **Review outputs**: Check that results are actionable and informative

## Common Pitfalls

### Avoid These Issues

1. **Vague instructions**: "Test the server" instead of specific actions
2. **Missing context**: Not providing enough information about expected environment
3. **Unrealistic expectations**: Asking for operations the MCP server can't perform
4. **No error handling**: Not accounting for common failure scenarios
5. **Too complex**: Trying to test everything in one prompt

### Best Practices

1. **One concept per prompt**: Focus on specific functionality
2. **Progressive complexity**: Start simple, build up
3. **Clear language**: Use direct, unambiguous instructions
4. **Measurable outcomes**: Define what success looks like
5. **Realistic scope**: Keep evaluations focused and achievable

## Example Prompt Templates

### Basic Functionality Test
```
Basic {{server_type}} Server Test:

1. [Simple operation to verify connectivity]
2. [Basic read/list operation]
3. [Slightly more complex operation]
4. [Error case testing]

Success criteria:
- All operations complete successfully
- Error cases return appropriate messages
- Results are properly formatted
```

### Comparison Test
```
{{server_a}} vs {{server_b}} Comparison:

Perform identical operations with both servers:

1. [Operation 1]
2. [Operation 2]
3. [Operation 3]

Compare and report:
- Performance differences
- Feature availability
- Output quality
- Error handling
```

### Performance Benchmark
```
{{server_type}} Performance Benchmark:

1. [Operation to measure - baseline]
2. [Same operation with larger dataset]
3. [Complex operation timing]
4. [Concurrent operation test]

Performance targets:
- Baseline operation: < X seconds
- Large dataset: < Y seconds
- Complex operation: < Z seconds
```

These templates help ensure consistent, thorough evaluation across different MCP server types and use cases.