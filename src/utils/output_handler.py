import os
from datetime import datetime
from pathlib import Path
import json

class WorkflowOutputHandler:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_output(self, query, filenames, response, workflow_name="workflow"):
        """
        Save workflow output as markdown with timestamp
        
        Args:
            query (str): The original query
            filenames (list): List of filenames involved
            response (str): The workflow response/result
            workflow_name (str): Name of the workflow for the filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{workflow_name}_{timestamp}.md"
        output_path = self.output_dir / output_filename
        
        content = f"""# Workflow Output - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Query
{query}

## Files Involved
{chr(10).join([f"- {filename}" for filename in filenames])}

## Response
{response}
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return output_path

    def get_latest_outputs(self, n=5):
        """Get the n most recent workflow outputs"""
        files = sorted(
            [f for f in self.output_dir.glob("*.md")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        return files[:n] 