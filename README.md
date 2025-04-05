# Document Research Agent

A tool for natural language querying of PDF documents using ChromaDB vector embeddings and LangChain for conversational AI.

## Overview

The Document Research Agent helps users query and analyze collections of PDF documents using natural language. It provides functionality to:

1. Process PDF documents and extract their content
2. Create vector embeddings for semantic search
3. Query documents using natural language questions
4. Generate detailed responses based on the document content

## Workflow Visualization

The following diagram shows the LangGraph workflow of the Document Research Agent:

![Document Research Agent Workflow](workflow_visualization.png)

## Document Ingestion

Before using the Document Research Agent, you need to ingest documents into the ChromaDB vector database:

```bash
# Install required packages
pip install PyPDF2

# Run the ingestion script (default path is test_data/)
python ingest_documents.py

# Ingest documents from a specific directory
python ingest_documents.py --directory path/to/your/pdfs
```

The ingestion script:
- Extracts text from PDF files
- Splits documents into chunks
- Creates embeddings using OpenAI's embedding model
- Stores documents and embeddings in ChromaDB

## Architecture

- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **LangChain**: Framework for building language model applications, with structured output support
- **LangGraph**: Flow control for conversational agents
- **OpenAI**: LLM API for natural language processing

## Features

- **Document Processing**: Automatically processes PDF documents and extracts text content
- **Semantic Search**: Retrieves the most relevant document sections based on natural language queries
- **Context-Aware Responses**: Generates responses that consider the document context
- **Command-Line Interface**: Simple CLI for interacting with the agent
- **Structured Output with LangChain**: Uses LangChain's structured output capabilities for reliable response formatting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/document_research_agent.git
cd document_research_agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by copying the template and adding your API keys:
```bash
cp .env.template .env
# Edit .env file with your OpenAI API key and other settings
```

## Usage

### Basic Query

```bash
python -m src.main "What are the key provisions in this contract?" --filenames path/to/document.pdf
```

### Using the Agent in Python Code

```python
from src.agent import DocumentResearchAgent

# Initialize the agent
agent = DocumentResearchAgent()

# Run a query
result = agent.run("What are the key provisions in this contract?", filenames=["path/to/document.pdf"])

# Access the results
print(result["final_answer"])
for citation in result["citations"]:
    print(f"- {citation['filename']}, Page {citation['page']}: '{citation['text']}'")
```

### Switching LLM Implementations

The Document Research Agent supports two LLM implementations:
1. The original OpenAI implementation
2. The LangChain implementation with structured output

Use the provided script to switch between implementations:

```bash
# Switch to the LangChain implementation
./switch_llm_implementation.py langchain

# Switch to the original OpenAI implementation
./switch_llm_implementation.py openai
```

### Check Collection Status

```bash
python -m src.main "Check collection status" --filenames path/to/document.pdf --check-collection
```

### Run with Verbose Output

```bash
python -m src.main "What are the key provisions in this contract?" --filenames path/to/document.pdf --verbose
```

## Evaluation Framework

The repository includes a comprehensive evaluation framework for testing and benchmarking the agent's performance. See [EVALUATION_README.md](EVALUATION_README.md) for details.

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_chroma_retriever.py
```

### Test Documents

The `test_data` directory contains sample legal documents for testing the agent. You can use these to evaluate performance with different document types.

### Evaluation Script

The `test_documents.py` script automates testing across different document types:

```bash
# Run all sample queries
./test_documents.py

# Run a specific query
./test_documents.py --query-index 2
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 