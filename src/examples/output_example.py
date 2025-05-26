from src.utils.output_handler import WorkflowOutputHandler

def main():
    # Initialize the output handler
    output_handler = WorkflowOutputHandler()
    
    # Example workflow data
    query = "How do I process PDF documents?"
    filenames = ["document_processor.py", "pdf_extractor.py"]
    response = """
The document processing workflow consists of the following steps:
1. Extract text from PDF using pdf_extractor.py
2. Process the extracted text using document_processor.py
3. Store results in the database
    """
    
    # Save the output
    output_path = output_handler.save_output(
        query=query,
        filenames=filenames,
        response=response,
        workflow_name="document_processing"
    )
    print(f"Output saved to: {output_path}")
    
    # Get recent outputs
    recent_outputs = output_handler.get_latest_outputs(n=3)
    print("\nRecent outputs:")
    for output_file in recent_outputs:
        print(f"- {output_file}")

if __name__ == "__main__":
    main() 