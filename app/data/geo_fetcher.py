# app/data/geo_fetcher.py
import pandas as pd
import numpy as np
import os
import tempfile
import GEOparse
import streamlit as st
from typing import Dict, Any, Optional

class GEOFetcher:
    """Class for fetching and processing data from NCBI GEO repository."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the GEO fetcher with optional caching directory."""
        self.cache_dir = cache_dir or tempfile.gettempdir()
        
    def fetch_geo_dataset(self, geo_accession: str) -> Dict[str, Any]:
        """Fetch dataset from GEO using the provided accession number."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Use GEOparse to fetch the dataset
            gse = GEOparse.get_GEO(geo=geo_accession, destdir=self.cache_dir)
            
            # Extract basic information
            title = gse.metadata.get("title", ["Unknown"])[0] if gse.metadata.get("title") else "Unknown"
            summary = gse.metadata.get("summary", ["No summary available"])[0] if gse.metadata.get("summary") else "No summary available"
            
            # Process sample metadata
            sample_metadata = {}
            for gsm_name, gsm in gse.gsms.items():
                characteristics = gsm.metadata.get("characteristics_ch1", [])
                source_name = gsm.metadata.get("source_name_ch1", ["Unknown"])[0]
                title_sample = gsm.metadata.get("title", ["Unknown"])[0]
                
                sample_metadata[gsm_name] = {
                    "title": title_sample,
                    "source": source_name,
                    "characteristics": characteristics
                }
            
            return {
                "gse_object": gse,
                "geo_accession": geo_accession,
                "title": title,
                "summary": summary,
                "sample_metadata": sample_metadata,
                "num_samples": len(gse.gsms),
                "platforms": list(gse.gpls.keys()) if gse.gpls else []
            }
            
        except Exception as e:
            raise Exception(f"Failed to fetch GEO dataset {geo_accession}: {str(e)}")
    
    def process_expression_data(self, geo_data: Dict[str, Any]) -> pd.DataFrame:
        """Process and combine expression data from all samples."""
        try:
            gse = geo_data["gse_object"]
            
            # Initialize list to store expression data
            expression_matrices = []
            sample_names = []
            
            # Process each sample
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table') and gsm.table is not None:
                    # Get the expression table
                    table = gsm.table.copy()
                    
                    # Ensure we have the required columns
                    if 'ID_REF' in table.columns and 'VALUE' in table.columns:
                        # Create a clean expression series for this sample
                        expression_series = table.set_index('ID_REF')['VALUE']
                        expression_series.name = gsm_name
                        
                        # Convert to numeric, handling any non-numeric values
                        expression_series = pd.to_numeric(expression_series, errors='coerce')
                        
                        expression_matrices.append(expression_series)
                        sample_names.append(gsm_name)
            
            if not expression_matrices:
                raise ValueError("No valid expression data found in any samples")
            
            # Combine all expression data into a single DataFrame
            expression_df = pd.concat(expression_matrices, axis=1, join='outer')
            
            # Remove rows with all NaN values
            expression_df = expression_df.dropna(how='all')
            
            # Fill remaining NaN values with 0 (or you could use median/mean)
            expression_df = expression_df.fillna(0)
            
            return expression_df
            
        except Exception as e:
            raise Exception(f"Failed to process expression data: {str(e)}")
    
    def extract_sample_groups(self, geo_data: Dict[str, Any]) -> Dict[str, list]:
        """Extract sample groups based on characteristics."""
        try:
            sample_metadata = geo_data["sample_metadata"]
            groups = {"Control": [], "Disease": [], "Unknown": []}
            
            for sample_id, metadata in sample_metadata.items():
                characteristics = metadata.get("characteristics", [])
                title = metadata.get("title", "").lower()
                source = metadata.get("source", "").lower()
                
                # Simple classification based on common keywords
                is_control = any(
                    keyword in " ".join(characteristics + [title, source]).lower()
                    for keyword in ["control", "normal", "healthy", "wildtype", "wt"]
                )
                
                is_disease = any(
                    keyword in " ".join(characteristics + [title, source]).lower()
                    for keyword in ["disease", "cancer", "tumor", "patient", "affected", "mutant"]
                )
                
                if is_control and not is_disease:
                    groups["Control"].append(sample_id)
                elif is_disease and not is_control:
                    groups["Disease"].append(sample_id)
                else:
                    groups["Unknown"].append(sample_id)
            
            # Remove empty groups
            groups = {k: v for k, v in groups.items() if v}
            
            return groups
            
        except Exception as e:
            raise Exception(f"Failed to extract sample groups: {str(e)}")
    
    def get_dataset_info(self, geo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        try:
            gse = geo_data["gse_object"]
            
            # Platform information
            platform_info = {}
            if gse.gpls:
                for gpl_id, gpl in gse.gpls.items():
                    platform_info[gpl_id] = {
                        "title": gpl.metadata.get("title", ["Unknown"])[0] if gpl.metadata.get("title") else "Unknown",
                        "organism": gpl.metadata.get("organism", ["Unknown"])[0] if gpl.metadata.get("organism") else "Unknown",
                        "technology": gpl.metadata.get("technology", ["Unknown"])[0] if gpl.metadata.get("technology") else "Unknown"
                    }
            
            return {
                "title": geo_data["title"],
                "summary": geo_data["summary"],
                "num_samples": geo_data["num_samples"],
                "platforms": platform_info,
                "submission_date": gse.metadata.get("submission_date", ["Unknown"])[0] if gse.metadata.get("submission_date") else "Unknown",
                "last_update": gse.metadata.get("last_update_date", ["Unknown"])[0] if gse.metadata.get("last_update_date") else "Unknown"
            }
            
        except Exception as e:
            raise Exception(f"Failed to get dataset info: {str(e)}")
