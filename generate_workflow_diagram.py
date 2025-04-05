#!/usr/bin/env python3
"""
Script to generate a Mermaid diagram and PNG for the Document Research Agent workflow.
"""

import os
import requests
import base64
import subprocess
try:
    from src.graph_builder import build_graph
except ImportError:
    # If we can't import directly, create a simple workflow diagram
    build_graph = None

def create_mermaid_diagram():
    """Create a Mermaid diagram for the LangGraph workflow."""
    if build_graph:
        # Use the actual graph from the project
        try:
            graph = build_graph()
            mermaid_code = graph.get_graph().draw_mermaid()
            return mermaid_code
        except Exception as e:
            print(f"Error generating diagram from actual graph: {e}")
            # Fall back to static diagram
    
    # Define a static Mermaid diagram if we can't generate from code
    return """
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([Start]):::first
    generate_search_queries(Generate Search Queries)
    retrieve_context(Retrieve Context)
    grade_context(Grade Context)
    generate_answer(Generate Answer)
    format_output(Format Output)
    __end__([End]):::last
    
    __start__ --> generate_search_queries;
    generate_search_queries --> retrieve_context;
    retrieve_context --> grade_context;
    grade_context -->|Context Sufficient| generate_answer;
    grade_context -->|Need More Information| generate_search_queries;
    generate_answer --> format_output;
    format_output --> __end__;
    
    classDef first fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#0277bd;
    classDef last fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32;
    classDef default fill:#f5f5f5,stroke:#757575,stroke-width:1px,color:#424242;
"""

def save_mermaid_to_file(mermaid_code, filename="workflow.mermaid"):
    """Save the Mermaid code to a file."""
    with open(filename, 'w') as f:
        f.write(mermaid_code)
    print(f"Saved Mermaid diagram to {filename}")
    return filename

def generate_png_via_api(mermaid_code, output_file="workflow.png"):
    """Generate a PNG from Mermaid code using the Mermaid.ink API."""
    # Encode the Mermaid code for URL
    encoded_diagram = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    
    # Use the Mermaid.ink API to generate the PNG
    url = f"https://mermaid.ink/img/{encoded_diagram}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Generated PNG diagram at {output_file}")
            return output_file
        else:
            print(f"Error generating PNG: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error accessing Mermaid.ink API: {e}")
        return None

def generate_png_via_mmdc(mermaid_file, output_file="workflow.png"):
    """Generate a PNG from Mermaid file using the mmdc CLI tool if available."""
    try:
        # Check if mmdc is installed
        subprocess.run(["mmdc", "--version"], capture_output=True, check=True)
        
        # Generate PNG
        subprocess.run([
            "mmdc", 
            "-i", mermaid_file, 
            "-o", output_file,
            "-b", "transparent"
        ], check=True)
        
        print(f"Generated PNG diagram at {output_file}")
        return output_file
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error using mmdc CLI tool: {e}")
        return None

def main():
    """Generate the workflow diagram."""
    # Create diagrams directory if it doesn't exist
    os.makedirs("diagrams", exist_ok=True)
    
    # Create the Mermaid diagram
    mermaid_code = create_mermaid_diagram()
    
    # Save Mermaid code to file
    mermaid_file = save_mermaid_to_file("diagrams/workflow.mermaid")
    
    # Try to generate PNG via mmdc CLI tool
    png_file = generate_png_via_mmdc(mermaid_file, "diagrams/workflow.png")
    
    # If mmdc fails, try the API approach
    if not png_file:
        png_file = generate_png_via_api(mermaid_code, "diagrams/workflow.png")
    
    if png_file:
        # Add the diagram to the README
        print(f"\nTo include the diagram in your README.md, add the following line:")
        print(f"\n![Document Research Agent Workflow](diagrams/workflow.png)\n")
    else:
        # Provide instructions for manual generation
        print("\nCould not generate PNG automatically.")
        print("You can manually convert the Mermaid diagram using:")
        print("1. Visit https://mermaid.live")
        print("2. Paste the contents of diagrams/workflow.mermaid")
        print("3. Download the PNG and save it to diagrams/workflow.png")

if __name__ == "__main__":
    main() 