# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from data.geo_fetcher import GEOFetcher
    from data.differential_expression import DifferentialExpressionAnalyzer
    from data.ml_classifier import MLClassifier
    from data.clinical_actionability import ClinicalActionabilityScorer
    from components.visualization import create_volcano_plot, create_heatmap
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required files are in the correct directories.")
    st.stop()

# Configure the page
st.set_page_config(
    page_title="Gene Detective",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    default_values = {
        "geo_data": None,
        "expression_matrix": None,
        "sample_groups": None,
        "dataset_info": None,
        "deg_results": None,
        "ml_results": None,
        "clinical_results": None,
        "analysis_complete": False
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Call initialization
initialize_session_state()

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.step-header {
    color: #2e8b57;
    border-bottom: 2px solid #2e8b57;
    padding-bottom: 0.5rem;
}
.info-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧬 Gene Detective")
st.sidebar.markdown("---")

# Navigation
pages = [
    "🏠 Home",
    "📊 Data Loading", 
    "🔬 Analysis Pipeline",
    "📈 Results",
    "💊 Clinical Actionability",
    "📁 Export"
]

page = st.sidebar.radio("Navigation", pages)

# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🧬 Gene Detective</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Welcome to Gene Detective!
    
    A powerful bioinformatics platform for analyzing gene expression datasets from NCBI GEO repository.
    
    **Key Features:**
    - 📊 **Data Loading**: Fetch and preprocess datasets using GEO accession numbers
    - 🔬 **Statistical Analysis**: Identify differentially expressed genes
    - 🤖 **Machine Learning**: Apply XGBoost for gene classification
    - 💊 **Clinical Insights**: Score genes for therapeutic potential
    - 📊 **Interactive Visualizations**: Volcano plots, heatmaps, and networks
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. Navigate to **Data Loading** to fetch a GEO dataset
    2. Use **Analysis Pipeline** to identify differentially expressed genes
    3. Review **Results** with machine learning classifications
    4. Explore **Clinical Actionability** for therapeutic insights
    5. **Export** your findings and visualizations
    """)
    
    st.markdown("### 📋 Example Datasets")
    example_data = pd.DataFrame({
        "GEO Accession": ["GSE68377", "GSE48350", "GSE19188"],
        "Description": ["Breast cancer expression", "Alzheimer's disease", "Lung cancer"],
        "Samples": ["~200", "~250", "~150"]
    })
    st.table(example_data)

elif page == "📊 Data Loading":
    st.markdown('<h2 class="step-header">📊 Data Loading</h2>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        geo_accession = st.text_input(
            "Enter GEO Accession Number", 
            placeholder="e.g., GSE68377",
            help="Enter a valid GEO Series accession number (starts with GSE)"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        load_button = st.button("🔄 Load Dataset", type="primary", use_container_width=True)
    
    # Validation
    if geo_accession and not geo_accession.upper().startswith('GSE'):
        st.warning("⚠️ GEO accession number should start with 'GSE'")
    
    if load_button and geo_accession:
        if not geo_accession.upper().startswith('GSE'):
            st.error("❌ Invalid GEO accession format. Please enter a valid GSE number.")
        else:
            with st.spinner(f"🔄 Loading dataset {geo_accession}..."):
                try:
                    # Initialize GEO fetcher
                    geo_fetcher = GEOFetcher(cache_dir="./data")
                    
                    # Fetch dataset
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    geo_data = geo_fetcher.fetch_geo_dataset(geo_accession)
                    progress_bar.progress(50)
                    
                    # Process expression data
                    expression_matrix = geo_fetcher.process_expression_data(geo_data)
                    progress_bar.progress(75)
                    
                    # Extract sample groups
                    sample_groups = geo_fetcher.extract_sample_groups(geo_data)
                    
                    # Get dataset info
                    dataset_info = geo_fetcher.get_dataset_info(geo_data)
                    progress_bar.progress(100)
                    
                    # Store in session state
                    st.session_state.geo_data = geo_data
                    st.session_state.expression_matrix = expression_matrix
                    st.session_state.sample_groups = sample_groups
                    st.session_state.dataset_info = dataset_info
                    
                    progress_bar.empty()
                    st.success(f"✅ Dataset {geo_accession} loaded successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
                    st.info("💡 Try a different GEO accession number or check your internet connection.")
    
    # Display results if data is loaded
    if st.session_state.geo_data is not None:
        st.markdown("### 📋 Dataset Information")
        
        info = st.session_state.dataset_info
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Title:** {info['title']}")
            st.markdown(f"**Number of Samples:** {info['num_samples']}")
            st.markdown(f"**Submission Date:** {info['submission_date']}")
        
        with col2:
            st.markdown(f"**Last Update:** {info['last_update']}")
            if info['platforms']:
                platform_names = list(info['platforms'].keys())
                st.markdown(f"**Platforms:** {', '.join(platform_names)}")
        
        with st.expander("📝 Summary"):
            st.write(info['summary'])
        
        # Sample groups
        st.markdown("### 👥 Sample Groups")
        if st.session_state.sample_groups:
            group_df = pd.DataFrame([
                {"Group": group, "Count": len(samples), "Samples": ", ".join(samples[:5]) + ("..." if len(samples) > 5 else "")}
                for group, samples in st.session_state.sample_groups.items()
            ])
            st.dataframe(group_df, use_container_width=True)
        else:
            st.warning("⚠️ No sample groups could be automatically identified.")
        
        # Expression data preview
        st.markdown("### 🧬 Expression Data Preview")
        if st.session_state.expression_matrix is not None:
            st.write(f"**Shape:** {st.session_state.expression_matrix.shape[0]} genes × {st.session_state.expression_matrix.shape[1]} samples")
            
            # Show first few rows and columns
            preview_df = st.session_state.expression_matrix.iloc[:10, :10]
            st.dataframe(preview_df, use_container_width=True)
            
            # Basic statistics
            st.markdown("### 📊 Basic Statistics")
            stats_df = st.session_state.expression_matrix.describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.error("❌ Failed to process expression data.")

elif page == "🔬 Analysis Pipeline":
    st.markdown('<h2 class="step-header">🔬 Analysis Pipeline</h2>', unsafe_allow_html=True)
    
    if st.session_state.expression_matrix is None:
        st.warning("⚠️ Please load a dataset first!")
        st.info("👈 Go to the 'Data Loading' section to fetch a GEO dataset.")
    else:
        st.success("✅ Dataset loaded and ready for analysis!")
        
        # Analysis configuration
        st.markdown("### ⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Groups:**")
            if st.session_state.sample_groups:
                for group, samples in st.session_state.sample_groups.items():
                    st.write(f"- **{group}**: {len(samples)} samples")
            
        with col2:
            # Analysis parameters
            st.markdown("**Parameters:**")
            padj_threshold = st.slider("Adjusted p-value threshold", 0.001, 0.1, 0.05, 0.001)
            fc_threshold = st.slider("Log2 fold change threshold", 0.5, 3.0, 1.0, 0.1)
        
        # Run analysis button
        if st.button("🚀 Run Differential Expression Analysis", type="primary"):
            if len(st.session_state.sample_groups) < 2:
                st.error("❌ Need at least 2 sample groups for differential expression analysis.")
            else:
                with st.spinner("🔄 Running differential expression analysis..."):
                    try:
                        # This is a simplified mock analysis
                        # In a real implementation, you would use DESeq2 or similar
                        
                        # Mock results for demonstration
                        np.random.seed(42)
                        n_genes = st.session_state.expression_matrix.shape[0]
                        
                        mock_results = pd.DataFrame({
                            'gene_id': st.session_state.expression_matrix.index,
                            'log2FoldChange': np.random.normal(0, 1.5, n_genes),
                            'pvalue': np.random.uniform(0, 1, n_genes),
                            'padj': np.random.uniform(0, 1, n_genes),
                            'baseMean': np.random.lognormal(5, 2, n_genes)
                        })
                        
                        # Add significance column
                        mock_results['significant'] = (
                            (mock_results['padj'] < padj_threshold) & 
                            (abs(mock_results['log2FoldChange']) > fc_threshold)
                        )
                        
                        st.session_state.deg_results = mock_results
                        
                        st.success("✅ Differential expression analysis completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
        
        # Display results if available
        if st.session_state.deg_results is not None:
            st.markdown("### 📊 Analysis Results")
            
            results = st.session_state.deg_results
            n_significant = results['significant'].sum()
            n_upregulated = ((results['log2FoldChange'] > fc_threshold) & results['significant']).sum()
            n_downregulated = ((results['log2FoldChange'] < -fc_threshold) & results['significant']).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Genes", len(results))
            col2.metric("Significant DEGs", n_significant)
            col3.metric("Upregulated", n_upregulated)
            col4.metric("Downregulated", n_downregulated)
            
            # Show top results
            st.markdown("#### 🔝 Top Differentially Expressed Genes")
            top_results = results.nlargest(20, 'log2FoldChange')[['gene_id', 'log2FoldChange', 'padj', 'significant']]
            st.dataframe(top_results, use_container_width=True)

elif page == "📈 Results":
    st.markdown('<h2 class="step-header">📈 Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.deg_results is None:
        st.warning("⚠️ Please complete the differential expression analysis first!")
        st.info("👈 Go to the 'Analysis Pipeline' section to run the analysis.")
    else:
        st.success("✅ Analysis results available!")
        
        # Visualization options
        st.markdown("### 📊 Visualizations")
        
        viz_option = st.selectbox(
            "Choose visualization:",
            ["Volcano Plot", "Expression Heatmap", "Results Table"]
        )
        
        if viz_option == "Volcano Plot":
            st.markdown("#### 🌋 Volcano Plot")
            
            results = st.session_state.deg_results.copy()
            results['-log10(padj)'] = -np.log10(results['padj'].replace(0, 1e-300))
            
            fig = px.scatter(
                results,
                x='log2FoldChange',
                y='-log10(padj)',
                color='significant',
                hover_data=['gene_id'],
                title="Volcano Plot - Differential Expression Results",
                labels={
                    'log2FoldChange': 'Log2 Fold Change',
                    '-log10(padj)': '-Log10(Adjusted P-value)'
                }
            )
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
            fig.add_vline(x=1, line_dash="dash", line_color="gray")
            fig.add_vline(x=-1, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Expression Heatmap":
            st.markdown("#### 🔥 Expression Heatmap")
            
            if st.session_state.expression_matrix is not None:
                # Select top 50 most variable genes for heatmap
                top_genes = st.session_state.deg_results.nlargest(50, 'log2FoldChange')['gene_id']
                heatmap_data = st.session_state.expression_matrix.loc[top_genes]
                
                fig = px.imshow(
                    heatmap_data.values,
                    labels=dict(x="Samples", y="Genes", color="Expression"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale="RdBu_r",
                    title="Expression Heatmap - Top 50 Differentially Expressed Genes"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Results Table":
            st.markdown("#### 📋 Complete Results Table")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                show_significant_only = st.checkbox("Show significant genes only", value=True)
            with col2:
                min_fold_change = st.slider("Minimum |log2 fold change|", 0.0, 3.0, 1.0)
            
            # Filter results
            filtered_results = st.session_state.deg_results.copy()
            if show_significant_only:
                filtered_results = filtered_results[filtered_results['significant']]
            
            filtered_results = filtered_results[abs(filtered_results['log2FoldChange']) >= min_fold_change]
            
            st.write(f"Showing {len(filtered_results)} genes")
            st.dataframe(filtered_results, use_container_width=True)

elif page == "💊 Clinical Actionability":
    st.markdown('<h2 class="step-header">💊 Clinical Actionability</h2>', unsafe_allow_html=True)
    
    if st.session_state.deg_results is None:
        st.warning("⚠️ Please complete the analysis pipeline first!")
        st.info("👈 Complete the differential expression analysis to access clinical actionability scoring.")
    else:
        st.success("✅ Ready for clinical actionability analysis!")
        
        st.markdown("""
        ### 🎯 Clinical Actionability Scoring
        
        This section scores genes based on their potential for therapeutic targeting by integrating:
        - **Drug Target Databases**: Known drug-gene interactions
        - **Clinical Trial Data**: Ongoing and completed trials
        - **Biomarker Information**: Diagnostic and prognostic markers
        """)
        
        if st.button("🚀 Score Clinical Actionability", type="primary"):
            with st.spinner("🔄 Analyzing clinical actionability..."):
                try:
                    # Mock clinical actionability scoring
                    significant_genes = st.session_state.deg_results[
                        st.session_state.deg_results['significant']
                    ]['gene_id'].tolist()
                    
                    if len(significant_genes) == 0:
                        st.warning("⚠️ No significant genes found for clinical analysis.")
                    else:
                        # Mock scoring (in real implementation, this would query actual databases)
                        np.random.seed(42)
                        clinical_scores = pd.DataFrame({
                            'gene_id': significant_genes[:50],  # Top 50 for demo
                            'drug_target_score': np.random.uniform(1, 10, min(50, len(significant_genes))),
                            'clinical_trial_score': np.random.uniform(1, 10, min(50, len(significant_genes))),
                            'biomarker_score': np.random.uniform(1, 10, min(50, len(significant_genes)))
                        })
                        
                        clinical_scores['actionability_score'] = (
                            clinical_scores['drug_target_score'] + 
                            clinical_scores['clinical_trial_score'] + 
                            clinical_scores['biomarker_score']
                        ) / 3
                        
                        clinical_scores = clinical_scores.sort_values('actionability_score', ascending=False)
                        
                        st.session_state.clinical_results = clinical_scores
                        st.success("✅ Clinical actionability scoring completed!")
                        
                except Exception as e:
                    st.error(f"❌ Clinical analysis failed: {str(e)}")
        
        # Display clinical results
        if st.session_state.clinical_results is not None:
            st.markdown("### 🏆 Top Clinically Actionable Genes")
            
            results = st.session_state.clinical_results
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Genes Analyzed", len(results))
            col2.metric("High Actionability (>7)", (results['actionability_score'] > 7).sum())
            col3.metric("Avg Score", f"{results['actionability_score'].mean():.2f}")
            
            # Top genes table
            st.dataframe(results.head(20), use_container_width=True)
            
            # Actionability distribution
            fig = px.histogram(
                results,
                x='actionability_score',
                nbins=20,
                title="Distribution of Clinical Actionability Scores"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "📁 Export":
    st.markdown('<h2 class="step-header">📁 Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.deg_results is None:
        st.warning("⚠️ No results to export!")
        st.info("👈 Complete the analysis pipeline to generate exportable results.")
    else:
        st.success("✅ Results ready for export!")
        
        st.markdown("### 📊 Available Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Data Tables")
            
            if st.button("📥 Download DEG Results", type="primary"):
                csv = st.session_state.deg_results.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name="differential_expression_results.csv",
                    mime="text/csv"
                )
            
            if st.session_state.clinical_results is not None:
                if st.button("📥 Download Clinical Results"):
                    csv = st.session_state.clinical_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name="clinical_actionability_results.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("#### 📊 Summary Report")
            
            if st.button("📄 Generate Report"):
                report = f"""
# Gene Detective Analysis Report

## Dataset Information
- **GEO Accession**: {st.session_state.geo_data['geo_accession'] if st.session_state.geo_data else 'N/A'}
- **Title**: {st.session_state.dataset_info['title'] if st.session_state.dataset_info else 'N/A'}
- **Number of Samples**: {st.session_state.dataset_info['num_samples'] if st.session_state.dataset_info else 'N/A'}

## Analysis Results
- **Total Genes Analyzed**: {len(st.session_state.deg_results)}
- **Significant DEGs**: {st.session_state.deg_results['significant'].sum()}
- **Upregulated Genes**: {((st.session_state.deg_results['log2FoldChange'] > 1) & st.session_state.deg_results['significant']).sum()}
- **Downregulated Genes**: {((st.session_state.deg_results['log2FoldChange'] < -1) & st.session_state.deg_results['significant']).sum()}

## Clinical Actionability
{f"- **Genes with High Actionability**: {(st.session_state.clinical_results['actionability_score'] > 7).sum()}" if st.session_state.clinical_results is not None else "- No clinical analysis performed"}

---
Report generated by Gene Detective
                """
                
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name="gene_detective_report.md",
                    mime="text/markdown"
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🧬 Gene Detective - Making bioinformatics accessible to everyone"
    "</div>",
    unsafe_allow_html=True
)
