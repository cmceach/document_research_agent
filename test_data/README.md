# Document Research Test Dataset

This directory contains a collection of legal documents for testing the Document Research Agent.

## Documents in the Collection

The collection consists of:
- 10 generated legal documents using reportlab
- 4 external sample documents

For a detailed index of the documents, see [document_index.md](document_index.md).

## How to Use the Test Data

You can use these documents to test the Document Research Agent in several ways:

### 1. Direct Usage

```bash
python -m src.main "What are the confidentiality provisions in the agreements?" --filenames test_data/legal_document_03_non_disclosure_agreement.pdf test_data/sample_nda.pdf
```

### 2. Using the Test Script

We've created a test script that contains predefined queries for various document types:

```bash
# Run all sample queries
./test_documents.py

# Run with verbose output
./test_documents.py --verbose

# Run a specific query by index (0-5)
./test_documents.py --query-index 2

# Check collection status instead of running queries
./test_documents.py --check-collection
```

### 3. Testing Collection Status

To check if the documents are properly loaded in the ChromaDB collection:

```bash
python -m src.main "Test query" --filenames test_data/legal_document_01_employment_contract.pdf --check-collection
```

## Generate More Test Documents

If you need more test documents, you can use the document generation script:

```bash
python create_sample_docs.py
```

You can modify the templates and sample data in the script to generate different types of documents.

## Document Types

The test collection includes a variety of legal document types:
- Employment contracts
- Service agreements
- Non-disclosure agreements
- Lease agreements
- Purchase agreements
- Shuttle service contract 