import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


# Transformation Functions
def negative_exponential(x, alpha, beta):
    return 1 - np.exp(-alpha * x**beta)


def indexed_exponential(x, alpha, beta):
    return 1 - np.exp(-alpha * (x / np.max(x))**beta)


def s_shape(x, alpha, beta):
    return 1 / (1 + np.exp(-alpha * (x - beta)))


def s_origin(x, alpha, beta):
    return 1 / (1 + np.exp(-alpha * (x - beta))) - 0.5


# App Header
st.title("VizionEDA")
st.write(
    "Analyze activity and spend data with transformations, filters, and visualizations."
)

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    # Load and preview data
    df = pd.read_csv(uploaded_file)
    st.write("## Uploaded Data Preview")
    st.dataframe(df.head())

    # Automatic Identification of Activity and Spend Variables
    media_keywords = [
        "Impressions", "Clicks", "Search", "OLV", "OOH", "Display",
        "Circulation", "Magazine", "Newspaper", "Social"
    ]
    spend_keywords = ["Spend", "Cost"]

    media_vars = [
        col for col in df.columns
        if any(keyword in col for keyword in media_keywords) and not any(
            keyword in col for keyword in spend_keywords)
    ]
    spend_vars = [
        col for col in df.columns
        if any(keyword in col for keyword in spend_keywords)
    ]

    st.write("### Identified Media and Spend Variables")
    st.write("Media Variables:", media_vars)
    st.write("Spend Variables:", spend_vars)

    # Global Filters
    st.sidebar.write("## Global Filters")
    filters = {}

    # Dynamically identify columns and populate filter options
    filter_columns = ["Geography", "Product", "Campaign", "Outlet", "Creative"]
    for col_name in filter_columns:
        col_name_lower = col_name.lower()
        if col_name_lower in df.columns:
            unique_values = df[col_name_lower].dropna().unique()
            selected_value = st.sidebar.selectbox(
                f"Filter by {col_name}", ["All"] + list(unique_values))
            filters[col_name_lower] = selected_value

    # Apply filters to the DataFrame
    filtered_df = df.copy()
    for col, value in filters.items():
        if value != "All":
            filtered_df = filtered_df[filtered_df[col] == value]

    # Univariate Analysis
    st.write("## Univariate Analysis")
    variable = st.selectbox("Select a variable for univariate analysis:",
                            filtered_df.columns)
    alpha = st.slider("Select alpha (α):", 0.01, 1.0, 0.1)
    beta = st.slider("Select beta (β):", 0.01, 2.0, 1.0)
    transformation = st.radio(
        "Select Transformation",
        ["Negative Exponential", "Indexed Exponential", "S-Shape", "S-Origin"])

    if variable:
        try:
            if transformation == "Negative Exponential":
                filtered_df[f"Transformed_{variable}"] = negative_exponential(
                    filtered_df[variable], alpha, beta)
            elif transformation == "Indexed Exponential":
                filtered_df[f"Transformed_{variable}"] = indexed_exponential(
                    filtered_df[variable], alpha, beta)
            elif transformation == "S-Shape":
                filtered_df[f"Transformed_{variable}"] = s_shape(
                    filtered_df[variable], alpha, beta)
            elif transformation == "S-Origin":
                filtered_df[f"Transformed_{variable}"] = s_origin(
                    filtered_df[variable], alpha, beta)

            st.write(f"## Transformed Data: {variable}")
            st.dataframe(filtered_df[[variable,
                                      f"Transformed_{variable}"]].head())

            # Visualization
            st.write(f"### Univariate Analysis: {variable}")
            fig, ax = plt.subplots()
            ax.plot(filtered_df[variable], label="Original", marker="o")
            ax.plot(filtered_df[f"Transformed_{variable}"],
                    label="Transformed",
                    marker="o")
            ax.set_title(f"Transformation of {variable}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error transforming data for variable '{variable}': {e}")

    # Multivariate Analysis
    st.write("## Multivariate Analysis")
    selected_vars = st.multiselect(
        "Select variables for multivariate analysis:", filtered_df.columns)
    sum_vars = st.checkbox("Sum Selected Variables for Analysis")

    if selected_vars:
        try:
            selected_data = filtered_df[selected_vars].copy()

            # Convert non-numeric columns to numeric representations
            for col in selected_data.select_dtypes(
                    include=['object', 'category']):
                selected_data[col] = selected_data[col].astype(str).apply(
                    lambda x: sum(ord(char) for char in x)
                    if isinstance(x, str) else 0)

            if sum_vars:
                selected_data["Sum_Variables"] = selected_data.sum(axis=1)
                transformed_sum = negative_exponential(
                    selected_data["Sum_Variables"], alpha, beta)
                st.write("### Transformation of Summed Variables")
                fig, ax = plt.subplots()
                ax.plot(selected_data["Sum_Variables"],
                        label="Sum of Variables",
                        marker="o")
                ax.plot(transformed_sum, label="Transformed Sum", marker="o")
                ax.set_title("Multivariate Analysis: Sum of Variables")
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("### Correlation Matrix")
                st.dataframe(selected_data.corr())
        except Exception as e:
            st.error(f"Error during multivariate analysis: {e}")

    # Feature: Final Report Generation
    st.write("## Final Report")
    if st.button("Generate Report"):
        try:
            report_buffer = BytesIO()
            with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer,
                                     index=False,
                                     sheet_name='Filtered Data')
                if "Sum_Variables" in selected_data.columns:
                    selected_data.to_excel(writer,
                                           index=False,
                                           sheet_name='Multivariate Analysis')

            report_buffer.seek(0)
            st.download_button(
                label="Download Report as Excel",
                data=report_buffer,
                file_name="crm_analysis_report.xlsx",
                mime=
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")

    # Feature: Documentation Section
    st.sidebar.write("## Documentation")
    st.sidebar.info("""
        **Transformations:**
        - Negative Exponential: Exponential decay transformation.
        - Indexed Exponential: Adjusted exponential decay using max normalization.
        - S-Shape: Sigmoidal transformation with adjustable inflection point.
        - S-Origin: Sigmoidal transformation centered around zero.

        **Filters:** Apply global filters for Geography, Product, Campaign, Outlet, and Creative.

        **Reports:** Download the analysis as an Excel report.
        """)
