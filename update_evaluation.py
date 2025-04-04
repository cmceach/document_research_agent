#!/usr/bin/env python3
"""
Script to update the evaluation spreadsheet with answers from the generated report.
"""

import os
import sys
import pandas as pd
import re
import glob
from datetime import datetime

def extract_answers_from_report(report_file):
    """Extract answers from the generated report file."""
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Define pattern to extract question-answer pairs
    pattern = r"## (.*?)\n\n.*?\n\n((?:### Question \d+: (.*?)\n\n(.*?)\n\n---\n\n)+)"
    doc_matches = re.findall(pattern, content, re.DOTALL)
    
    answers = {}
    for doc_match in doc_matches:
        doc_type = doc_match[0]
        qa_content = doc_match[1]
        
        # Extract individual Q&A pairs
        qa_pattern = r"### Question \d+: (.*?)\n\n(.*?)\n\n---"
        qa_matches = re.findall(qa_pattern, qa_content, re.DOTALL)
        
        doc_answers = []
        for qa_match in qa_matches:
            question = qa_match[0]
            answer = qa_match[1].strip()
            doc_answers.append({"question": question, "answer": answer})
        
        answers[doc_type] = doc_answers
    
    return answers

def update_excel_with_answers(excel_file, answers):
    """Update the Excel evaluation spreadsheet with the answers."""
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    # Update answers in DataFrame
    for index, row in df.iterrows():
        doc_type = row['Document Type']
        question = row['Question']
        
        if doc_type in answers:
            for qa in answers[doc_type]:
                if qa['question'] == question:
                    df.at[index, 'Actual Answer'] = qa['answer']
                    break
    
    # Create backup of original
    backup_file = f"{os.path.splitext(excel_file)[0]}_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    df_original = pd.read_excel(excel_file)
    df_original.to_excel(backup_file, index=False)
    print(f"Created backup: {backup_file}")
    
    # Save updated file
    df.to_excel(excel_file, index=False)
    print(f"Updated {excel_file} with answers")
    
    return excel_file

def main():
    """Update the evaluation spreadsheet with answers from the most recent report."""
    # Find the Excel file
    excel_files = glob.glob("document_research_evaluation*.xlsx")
    if not excel_files:
        print("Error: No evaluation spreadsheet found")
        return 1
    
    excel_file = excel_files[0]
    
    # Find the most recent report file
    report_files = sorted(glob.glob("document_research_report_*.md"), reverse=True)
    if not report_files:
        print("Error: No report file found")
        return 1
    
    report_file = report_files[0]
    print(f"Using report file: {report_file}")
    
    # Extract answers from report
    answers = extract_answers_from_report(report_file)
    
    # Update Excel with answers
    update_excel_with_answers(excel_file, answers)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 