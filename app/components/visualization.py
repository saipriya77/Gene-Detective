# app/components/visualization.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def create_volcano_plot(deg_results):
    """Create a volcano plot from differential expression results."""
    # Create volcano plot with Plotly
    fig = px.scatter(
        deg_results,
        x="log2FoldChange",
        y="-log10(padj)",
        hover_name="gene_id",
        color="log2FoldChange",
        color_continuous_scale="RdBu_r",
        range_color=[-3, 3],
        title="Volcano Plot of Differential Expression Results",
    )
    
    # Add threshold lines
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=-1, line_dash="dash", line_color="gray")
    
    return fig

def create_heatmap(expression_data, gene_list=None, sample_groups=None):
    """Create a heatmap from expression data."""
    # Filter data if gene_list is provided
    if gene_list is not None:
        expression_data = expression_data.loc[expression_data.index.isin(gene_list)]
    
    # Create heatmap with Plotly
    fig = px.imshow(
        expression_data,
        labels=dict(x="Sample", y="Gene", color="Expression"),
        color_continuous_scale="RdBu_r",
        title="Gene Expression Heatmap",
    )
    
    return fig

def create_network_plot(gene_drug_data):
    """Create a network plot of gene-drug interactions."""
    # Create network plot with Plotly
    # This is a simplified implementation
    
    # Create nodes for genes and drugs
    nodes = []
    edges = []
    
    # Add gene nodes
    for gene in gene_drug_data["gene_id"].unique():
        nodes.append({
            "id": gene,
            "label": gene,
            "type": "gene"
        })
    
    # Add drug nodes and edges
    for _, row in gene_drug_data.iterrows():
        gene = row["gene_id"]
        for drug in row["drugs"]:
            if drug not in [node["id"] for node in nodes]:
                nodes.append({
                    "id": drug,
                    "label": drug,
                    "type": "drug"
                })
            
            edges.append({
                "source": gene,
                "target": drug,
                "weight": row["actionability_score"]
            })
    
    # Create network plot
    # This would typically be implemented using a network visualization library
    # For this example, we'll return a placeholder
    
    return nodes, edges
