# Document Research Agent

An intelligent agent that researches and answers questions about your documents using LLMs and semantic search.

## Quick Start

```bash
git clone https://github.com/yourusername/document_research.git
cd document_research
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env  # Add your API keys
```

**Ingest documents:**
```bash
python scripts/ingest_documents.py path/to/your/documents/*.pdf
```

**Query documents:**
```python
from src.agent import DocumentResearchAgent

agent = DocumentResearchAgent()
result = agent.run(
    query="What are the key provisions in this contract?",
    filenames=["doc1.pdf", "doc2.pdf"]
)
print(result["final_answer"])
```

## Features

- **Document Processing**: PDF text extraction and semantic search
- **Context-Aware Responses**: LLM-generated answers with citations
- **Structured Output**: Reliable response formatting using LangChain
- **Error Handling**: Comprehensive error handling and status reporting

## Usage

### Command Line
```bash
python -m src.main "What are the payment terms?" --filenames contract.pdf
```

### Running the FastAPI Server

To start the FastAPI server, run the following command from the project root:

```bash
python -m src.main --serve --host <your_host> --port <your_port>
```
For example, to run on localhost, port 8000:
```bash
python -m src.main --serve --host 0.0.0.0 --port 8000
```
This will make the API accessible, typically at `http://<your_host>:<your_port>`. The `/research` endpoint will be available for POST requests. Refer to `docs/API.md` for more details on the API.

### Python API
```python
# Basic usage
result = agent.run(query="Summarize the main points", filenames=["doc.pdf"])

# With options
result = agent.run(
    query="What are the key provisions?",
    filenames=["contract.pdf"],
    max_iterations=5,
    include_scratchpad=True
)

# Access results
if result["success"]:
    print(f"Answer: {result['final_answer']}")
    for citation in result["citations"]:
        print(f"- {citation['filename']}: {citation['text']}")
```

## Architecture

- **ChromaDB**: Vector database for document embeddings
- **LangChain**: Framework with structured output support
- **LangGraph**: Flow control for conversational agents
- **OpenAI**: LLM API for natural language processing

## Development

```bash
# Run tests
python -m pytest

# Test with sample queries
python test_document_research.py --run-samples

# Check document collection
python test_document_research.py --check-collection
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and run tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details. 