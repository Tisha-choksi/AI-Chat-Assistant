import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Union
import json
import io
import base64

class DataTools:
    def __init__(self):
        self.current_df = None
        self.supported_formats = ['.csv', '.xlsx', '.json']

    def load_data(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Load data from various file formats
        """
        try:
            file_ext = filename[filename.rfind('.'):].lower()
            
            if file_ext not in self.supported_formats:
                return {
                    "success": False,
                    "error": f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
                }

            if file_ext == '.csv':
                self.current_df = pd.read_csv(io.BytesIO(file_content))
            elif file_ext == '.xlsx':
                self.current_df = pd.read_excel(io.BytesIO(file_content))
            elif file_ext == '.json':
                self.current_df = pd.read_json(io.BytesIO(file_content))

            return {
                "success": True,
                "columns": list(self.current_df.columns),
                "shape": self.current_df.shape,
                "preview": self.current_df.head().to_dict()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Generate basic statistical analysis of the data
        """
        if self.current_df is None:
            return {"success": False, "error": "No data loaded"}

        try:
            numeric_cols = self.current_df.select_dtypes(include=[np.number]).columns
            stats = {
                "numeric_summary": self.current_df[numeric_cols].describe().to_dict(),
                "missing_values": self.current_df.isnull().sum().to_dict(),
                "column_types": self.current_df.dtypes.astype(str).to_dict()
            }

            return {
                "success": True,
                "statistics": stats
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def create_visualization(self, 
                           chart_type: str,
                           x_column: str,
                           y_column: str = None,
                           title: str = None) -> Dict[str, Any]:
        """
        Create various types of visualizations using Plotly
        """
        if self.current_df is None:
            return {"success": False, "error": "No data loaded"}

        try:
            fig = None
            if chart_type == "bar":
                fig = px.bar(self.current_df, x=x_column, y=y_column, title=title)
            elif chart_type == "line":
                fig = px.line(self.current_df, x=x_column, y=y_column, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(self.current_df, x=x_column, y=y_column, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(self.current_df, x=x_column, title=title)
            elif chart_type == "box":
                fig = px.box(self.current_df, y=y_column, title=title)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported chart type: {chart_type}"
                }

            # Convert plot to JSON for Streamlit
            plot_json = fig.to_json()

            return {
                "success": True,
                "plot": plot_json,
                "chart_type": chart_type
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_correlation(self, columns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze correlations between numeric columns
        """
        if self.current_df is None:
            return {"success": False, "error": "No data loaded"}

        try:
            numeric_df = self.current_df.select_dtypes(include=[np.number])
            
            if columns:
                numeric_df = numeric_df[columns]

            if numeric_df.empty:
                return {
                    "success": False,
                    "error": "No numeric columns available for correlation analysis"
                }

            correlation_matrix = numeric_df.corr()
            
            # Create correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu'
            ))
            
            fig.update_layout(title="Correlation Heatmap")

            return {
                "success": True,
                "correlation_matrix": correlation_matrix.to_dict(),
                "plot": fig.to_json()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate automated insights about the data
        """
        if self.current_df is None:
            return {"success": False, "error": "No data loaded"}

        try:
            insights = []
            
            # Basic dataset information
            insights.append(f"Dataset contains {self.current_df.shape[0]} rows and {self.current_df.shape[1]} columns")
            
            # Missing values analysis
            missing = self.current_df.isnull().sum()
            if missing.any():
                insights.append(f"Found missing values in columns: {', '.join(missing[missing > 0].index)}")
            
            # Numeric column analysis
            numeric_cols = self.current_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                insights.append(f"{col}: Mean = {self.current_df[col].mean():.2f}, Std = {self.current_df[col].std():.2f}")
            
            # Categorical column analysis
            categorical_cols = self.current_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_vals = self.current_df[col].nunique()
                insights.append(f"{col}: {unique_vals} unique values")

            return {
                "success": True,
                "insights": insights
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
