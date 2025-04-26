import streamlit as st
import plotly.graph_objects as go
import json
from typing import Dict, Any, List
import pandas as pd

def display_tool_output(output: Dict[str, Any], tool_type: str):
    """
    Display tool output in a formatted way using Streamlit
    """
    if not output["success"]:
        st.error(f"Error: {output['error']}")
        return

    if tool_type == "text_summary":
        st.write("📝 **Summary:**")
        st.write(output["summary"])
        st.info(f"Summary length: {output['length']} words")

    elif tool_type == "sentiment":
        st.write("😊 **Sentiment Analysis:**")
        sentiment = output["sentiment"]
        confidence = output["confidence"]
        
        # Display sentiment with emoji
        emoji = "😊" if sentiment == "POSITIVE" else "😢" if sentiment == "NEGATIVE" else "😐"
        st.write(f"{emoji} Sentiment: {sentiment}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence:.2%}")

    elif tool_type == "translation":
        st.write("🌐 **Translation:**")
        st.write(output["translation"])
        st.info(f"Target Language: {output['target_language']}")

    elif tool_type == "keywords":
        st.write("🔑 **Keywords:**")
        for keyword in output["keywords"]:
            st.markdown(f"- {keyword}")
        st.info(f"Found {output['count']} keywords")

    elif tool_type == "grammar":
        st.write("📚 **Grammar Analysis:**")
        st.write(output["analysis"])
        
    elif tool_type == "data_stats":
        st.write("📊 **Data Statistics:**")
        stats = output["statistics"]
        
        # Display numeric summary
        if "numeric_summary" in stats:
            st.write("Numeric Summary:")
            st.dataframe(pd.DataFrame(stats["numeric_summary"]))
        
        # Display missing values
        if "missing_values" in stats:
            st.write("Missing Values:")
            missing_df = pd.DataFrame.from_dict(stats["missing_values"], 
                                              orient='index', 
                                              columns=['Count'])
            st.dataframe(missing_df)
        
        # Display column types
        if "column_types" in stats:
            st.write("Column Types:")
            types_df = pd.DataFrame.from_dict(stats["column_types"], 
                                            orient='index', 
                                            columns=['Type'])
            st.dataframe(types_df)

    elif tool_type == "visualization":
        st.write("📈 **Visualization:**")
        fig = go.Figure(json.loads(output["plot"]))
        st.plotly_chart(fig)

    elif tool_type == "search":
        st.write("🔍 **Search Results:**")
        for idx, result in enumerate(output["results"], 1):
            st.markdown(f"**Result {idx}** (Score: {result['similarity_score']:.2f})")
            st.write(result["document"])
            st.markdown("---")

    elif tool_type == "qa":
        st.write("❓ **Question Answering:**")
        st.write(output["answer"])
        st.info("Context used: " + output["context_used"])

    elif tool_type == "topics":
        st.write("📑 **Topics:**")
        for topic in output["topics"]:
            st.markdown(f"- {topic}")
        st.info(f"Found {output['count']} topics")

def format_error(error: str) -> str:
    """
    Format error messages for display
    """
    return f"⚠️ Error: {error}"

def create_download_link(data: Any, filename: str) -> str:
    """
    Create a download link for data
    """
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        return f'data:text/csv;charset=utf-8,{csv}'
    elif isinstance(data, dict):
        return f'data:text/json;charset=utf-8,{json.dumps(data)}'
    else:
        return f'data:text/plain;charset=utf-8,{str(data)}'

def validate_file_upload(uploaded_file) -> Dict[str, Any]:
    """
    Validate uploaded files
    """
    if uploaded_file is None:
        return {
            "valid": False,
            "error": "No file uploaded"
        }

    allowed_types = ['csv', 'xlsx', 'json', 'txt']
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type not in allowed_types:
        return {
            "valid": False,
            "error": f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        }

    return {
        "valid": True,
        "file_type": file_type
    }
