# Development Guide

## Project Overview

The Document Research Agent is an AI-powered system that answers user questions based *only* on the content of specified documents. It uses a ReAct (Reasoning+Acting) pattern with LangGraph orchestration, ChromaDB for vector search, and OpenAI's LLM for natural language processing.

### Key Features
- **Accurate Information Retrieval**: Answers derived solely from specified documents
- **Verifiable Answers**: Precise citations with text snippets, page numbers, and filenames
- **Controlled Scope**: Responses strictly limited to provided document set
- **Iterative Search**: ReAct pattern with up to 5 search iterations

## Current Status

### ✅ Completed Components
- **Core Agent**: LangGraph-based workflow with ReAct pattern
- **ChromaDB Integration**: Vector search with filename filtering
- **LLM Integration**: OpenAI API with structured output support
- **Testing Framework**: Comprehensive test suite with sample queries
- **Evaluation System**: Automated evaluation with scoring framework
- **Documentation**: Complete API documentation and usage examples

### 🔄 Active Development Areas

#### Code Quality Improvements
- [x] Consolidate redundant test files
- [ ] Optimize prompt templates for better performance
- [ ] Improve error handling and logging
- [ ] Add performance monitoring and metrics

#### Feature Enhancements
- [ ] Multi-turn conversation support
- [ ] Advanced citation extraction (sub-snippets)
- [ ] Batch processing for multiple queries
- [ ] Custom embedding model support

## Architecture & Requirements

### Core Workflow
1. **Input Processing**: Accept `original_query` and `filenames` list
2. **Iterative Context Gathering**: Generate search queries → Vector search → Evaluate context
3. **Answer Synthesis**: Generate final answer with citations from retrieved context
4. **Output Formatting**: Return JSON with answer and citations

### Technical Requirements
- **Document Indexing**: ChromaDB collection with chunked documents (600 chars, 200 overlap)
- **Metadata**: Each chunk includes `text`, `filename`, `page_number`
- **Iteration Limit**: Maximum 5 search iterations per query
- **Citation Format**: `text`, `page`, `filename` for each citation
- **Failure Handling**: Return "Information not found in provided documents" when appropriate

### Dependencies
- `langchain`, `langgraph` for workflow orchestration
- `openai` for LLM API access
- `chromadb` for vector database
- `pytest` for testing framework

## Development Workflow

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository>
cd document_research
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env  # Add your API keys
```

### 2. Testing
```bash
# Run unit tests
python -m pytest

# Run integration tests
python tests/test_document_research.py --run-samples

# Check collection status
python tests/test_document_research.py --check-collection
```

### 3. Development Guidelines

#### Code Style
- Follow PEP 8 conventions
- Use type hints for all functions
- Add comprehensive docstrings
- Maintain test coverage above 80%

#### Testing Strategy
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance tests for large document collections
- Evaluation tests with sample queries

#### Documentation
- Update README.md for user-facing changes
- Update this guide for development changes
- Maintain API documentation in docstrings
- Document configuration options

### 4. Performance Optimization

#### Current Metrics
- Average response time: ~10-15 seconds
- Token usage: ~2000-5000 tokens per query
- Memory usage: ~200MB for typical collections

#### Optimization Targets
- Reduce response time to <10 seconds
- Optimize token usage through better prompts
- Implement caching for repeated queries
- Add streaming responses for real-time feedback

### 5. Implementation Checklist

#### Phase 1: Core Components ✅
- [x] ChromaRetriever class with filename filtering
- [x] LLM wrapper module with structured output
- [x] AgentState management
- [x] Graph node functions (generate queries, retrieve context, grade context, final answer)
- [x] LangGraph workflow definition

#### Phase 2: Integration & Testing ✅
- [x] Main script with command-line interface
- [x] Unit tests for all components
- [x] Integration tests for end-to-end functionality
- [x] Sample documents and test queries

#### Phase 3: Refinement ✅
- [x] Prompt engineering and optimization
- [x] Error handling and edge cases
- [x] Documentation and examples
- [x] Performance testing

#### Phase 4: Advanced Features 🔄
- [ ] Multi-turn conversation support
- [ ] Batch query processing
- [ ] Advanced citation extraction
- [ ] Custom embedding models
- [ ] Performance monitoring

## Deployment Considerations

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL_NAME=gpt-4o  # or preferred model
CHROMA_DB_PATH=./chroma_db     # database location
```

### Production Setup
- Use persistent ChromaDB storage
- Implement proper logging and monitoring
- Add rate limiting for API calls
- Configure backup and recovery procedures

## Success Metrics

- **Answer Relevance**: Percentage of answers rated as relevant and accurate
- **Citation Correctness**: Percentage of citations accurately pointing to source
- **Task Completion Rate**: Percentage of queries resolved vs. "Information not found"
- **Average Latency**: Time from request to response
- **API Efficiency**: Average search iterations per query (≤5)

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes and add tests**
4. **Run the test suite**: `python -m pytest`
5. **Update documentation** as needed
6. **Submit a pull request**

### Pull Request Guidelines
- Include clear description of changes
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass
- Follow code style guidelines

## Troubleshooting

### Common Issues

#### ChromaDB Collection Not Found
```bash
# Check if documents are ingested
python scripts/ingest_documents.py test_data/*.pdf
```

#### OpenAI API Errors
- Verify API key is set correctly
- Check rate limits and quotas
- Ensure model name is correct

#### Memory Issues
- Reduce chunk size in ingestion
- Limit number of search results
- Use streaming for large responses

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python -m src.main "your query" --verbose
``` 