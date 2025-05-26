# API Reference

## DocumentResearchAgent

The main class for interacting with the Document Research Agent.

### Initialization

```python
from src.agent import DocumentResearchAgent

agent = DocumentResearchAgent()
```

### Methods

#### `run(query, filenames, max_iterations=5, include_scratchpad=False)`

Execute a document research query.

**Parameters:**
- `query` (str): The question to answer based on the documents
- `filenames` (list): List of document filenames to search within
- `max_iterations` (int, optional): Maximum search iterations (default: 5)
- `include_scratchpad` (bool, optional): Include agent reasoning in output (default: False)

**Returns:**
- `dict`: Result dictionary with the following structure:
  ```python
  {
      "success": bool,
      "final_answer": str,
      "citations": [
          {
              "text": str,      # Citation text
              "page": int,     # Page number
              "filename": str  # Source filename
          }
      ],
      "iterations": int,
      "scratchpad": str  # Only if include_scratchpad=True
  }
  ```

### Usage Examples

#### Basic Usage

```python
from src.agent import DocumentResearchAgent

agent = DocumentResearchAgent()
result = agent.run(
    query="What are the key provisions in this contract?",
    filenames=["contract.pdf", "amendment.pdf"]
)

if result["success"]:
    print(f"Answer: {result['final_answer']}")
    print(f"Found {len(result['citations'])} citations")
else:
    print("Query failed or no information found")
```

#### Advanced Usage with Options

```python
result = agent.run(
    query="What are the payment terms and deadlines?",
    filenames=["service_agreement.pdf"],
    max_iterations=3,
    include_scratchpad=True
)

# Access detailed results
print(f"Answer: {result['final_answer']}")
print(f"Search iterations: {result['iterations']}")

# Display citations
for i, citation in enumerate(result["citations"], 1):
    print(f"Citation {i}:")
    print(f"  File: {citation['filename']}")
    print(f"  Page: {citation['page']}")
    print(f"  Text: {citation['text'][:100]}...")

# View agent reasoning (if enabled)
if result.get("scratchpad"):
    print(f"Agent reasoning:\n{result['scratchpad']}")
```

#### Error Handling

```python
try:
    result = agent.run(
        query="What is the termination clause?",
        filenames=["nonexistent.pdf"]
    )
    
    if not result["success"]:
        print("No information found in provided documents")
    
except Exception as e:
    print(f"Error during query execution: {e}")
```

## Command Line Interface

### Basic Usage

```bash
python -m src.main "Your question here" --filenames document1.pdf document2.pdf
```

### Options

- `--filenames`: Space-separated list of document filenames
- `--max-iterations`: Maximum search iterations (default: 5)
- `--include-scratchpad`: Include agent reasoning in output
- `--output-format`: Output format (json, text) (default: text)

### Examples

```bash
# Basic query
python -m src.main "What are the payment terms?" --filenames contract.pdf

# Multiple documents
python -m src.main "What are the key provisions?" --filenames contract.pdf amendment.pdf addendum.pdf

# With debugging information
python -m src.main "What is the termination clause?" --filenames agreement.pdf --include-scratchpad

# JSON output
python -m src.main "What are the deliverables?" --filenames sow.pdf --output-format json
```

## Environment Configuration

### Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL_NAME=gpt-4o  # or your preferred model
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=documents
```

### Optional Environment Variables

```bash
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-ada-002
LOG_LEVEL=INFO
MAX_ITERATIONS=5
```

## Document Ingestion

Before querying documents, they must be ingested into the ChromaDB collection.

### Using the Ingestion Script

```bash
# Ingest PDF documents
python scripts/ingest_documents.py path/to/documents/*.pdf

# Ingest specific files
python scripts/ingest_documents.py document1.pdf document2.pdf document3.pdf
```

### Programmatic Ingestion

```python
from src.retriever.chroma_retriever import ChromaRetriever

retriever = ChromaRetriever()
# Ingestion methods would be called here
# (See ingestion script for detailed implementation)
```

## Response Format

### Successful Response

```json
{
    "success": true,
    "final_answer": "The payment terms specify that invoices are due within 30 days of receipt...",
    "citations": [
        {
            "text": "Payment shall be made within thirty (30) days of invoice receipt.",
            "page": 3,
            "filename": "service_agreement.pdf"
        }
    ],
    "iterations": 2
}
```

### No Information Found

```json
{
    "success": true,
    "final_answer": "Information not found in provided documents",
    "citations": [],
    "iterations": 5
}
```

### Error Response

```json
{
    "success": false,
    "error": "Error description",
    "iterations": 0
}
```

## Best Practices

### Query Formulation

- **Be specific**: "What are the payment terms?" vs "Tell me about payments"
- **Use document terminology**: Match language used in your documents
- **Ask focused questions**: One concept per query for best results

### Document Selection

- **Relevant files only**: Include only documents likely to contain the answer
- **Reasonable file count**: 5-10 documents work best for performance
- **Consistent naming**: Use clear, descriptive filenames

### Performance Optimization

- **Limit iterations**: Use `max_iterations=3` for faster responses
- **Cache results**: Store frequently accessed answers
- **Batch queries**: Group related questions for efficiency 