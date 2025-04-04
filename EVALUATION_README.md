# Document Research Agent Evaluation Framework

This framework provides tools for evaluating the performance of the Document Research Agent on various types of legal documents.

## Components

1. **Test Documents Collection**
   - Located in `test_data/` directory
   - 10 generated legal documents (employment contracts, service agreements, NDAs, lease agreements, purchase agreements)
   - 4 external sample documents
   - See `test_data/document_index.md` for details

2. **Evaluation Spreadsheet**
   - Excel file with 30 evaluation questions (5 per document type)
   - Contains expected answers based on document content
   - Provides scoring framework for evaluating actual answers

3. **Sample Report Generation**
   - Simulates running questions through the Document Research Agent
   - Creates a markdown report with detailed answers

4. **Test Automation**
   - Provides scripts for running tests and updating evaluation data

## Setup

Make sure you have the required Python packages installed:

```bash
pip install pandas openpyxl
```

## Workflow

### 1. Prepare Test Documents

The test documents are already generated and available in the `test_data/` directory. You can generate additional test documents if needed:

```bash
python create_sample_docs.py
```

### 2. Generate Evaluation Spreadsheet

The evaluation spreadsheet contains questions and expected answers for each document type:

```bash
python document_evaluation.py
```

This creates `document_research_evaluation.xlsx` with all the evaluation criteria.

### 3. Generate Sample Answers (for testing)

To test the evaluation framework, you can generate sample answers:

```bash
python generate_sample_report.py
```

This creates a markdown report with sample answers for all questions.

### 4. Update Evaluation with Answers

After generating sample answers or running actual tests with the Document Research Agent, update the evaluation spreadsheet:

```bash
python update_evaluation.py
```

This script finds the most recent report and updates the evaluation spreadsheet with the answers.

### 5. Run Tests with the Document Research Agent

Use the test script to run specific queries against the Document Research Agent:

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

## Evaluation Criteria

The evaluation spreadsheet provides columns for:

1. **Document Type**: Category of legal document
2. **Files**: List of document filenames used for the question
3. **Question ID**: Unique identifier for each question
4. **Question**: The specific question to answer
5. **Expected Answer**: The ideal response based on document content
6. **Actual Answer**: The actual response from the Document Research Agent
7. **Score (1-5)**: Subjective rating of answer quality
8. **Notes**: Additional observations

## Scoring Guidelines

When evaluating answers, use the following 1-5 scale:

- **5**: Perfect - Answer is complete, accurate, and addresses all relevant aspects
- **4**: Good - Answer is mostly correct but missing minor details or context
- **3**: Acceptable - Answer contains correct information but is incomplete
- **2**: Poor - Answer contains some correct information but has significant omissions or errors
- **1**: Incorrect - Answer is factually wrong or unrelated to the question

## Adding New Document Types

To evaluate additional document types:

1. Add documents to the `test_data/` directory
2. Update `document_evaluation.py` with new document type and questions
3. Regenerate the evaluation spreadsheet
4. Update `test_documents.py` with relevant test queries 