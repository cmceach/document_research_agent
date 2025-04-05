#!/usr/bin/env python3
"""
Generate a visualization of the Document Research Agent workflow using LangGraph.
"""

import os
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Import our graph builder
from src.graph_builder import build_graph

def generate_workflow_visualization():
    """Generate a visualization of the workflow as a Mermaid PNG."""
    # Build the graph - the function doesn't have an 'execute' parameter
    graph = build_graph()
    
    # Get the underlying graph object for visualization
    graph_obj = graph.get_graph()
    
    # Generate the Mermaid PNG
    png_data = graph_obj.draw_mermaid_png(
        curve_style=CurveStyle.LINEAR,
        node_colors=NodeStyles(
            first="#e6f7ff",  # Light blue for start
            last="#e6ffed",   # Light green for end
            default="#fff5e6"  # Light orange for other nodes
        ),
        wrap_label_n_words=3,
        output_file_path="workflow_visualization.png",
        draw_method=MermaidDrawMethod.API,
        background_color="white",
        padding=20
    )
    
    print(f"Visualization saved to workflow_visualization.png")
    return "workflow_visualization.png"

if __name__ == "__main__":
    generate_workflow_visualization() 