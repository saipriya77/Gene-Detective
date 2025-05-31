# app/data/differential_expression.py
import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet  # Correct submodule
from pydeseq2.ds import DeseqStats     # Correct submodule


class DifferentialExpressionAnalyzer:
    """Class for differential expression analysis of RNA-seq data."""
    
    def __init__(self):
        """Initialize the differential expression analyzer."""
        pass
    
    def run_deseq2(self, counts_df, metadata_df, design_formula):
        """Run DESeq2 analysis on the provided count data."""
        try:
            # Create DeseqDataSet object
            dds = DeseqDataSet(
                counts=counts_df,
                metadata=metadata_df,
                design_formula=design_formula,
                n_cpus=1
            )
            
            # Run DESeq2 analysis
            dds.deseq2()
            
            # Create DeseqStats object and get results
            stat_res = DeseqStats(dds)
            results_df = stat_res.summary()
            
            return results_df
        except Exception as e:
            raise Exception(f"DESeq2 analysis failed: {str(e)}")
