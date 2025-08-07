import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Analysis of Pseudo-Labeled Data")

if st.session_state['combined'] is not None and st.session_state['pl'] is not None:
    combined_data = st.session_state['combined']
    pl_instance = st.session_state['pl']
    
    # Ensure target_col_for_analysis is retrieved correctly from the stored pl_instance
    target_col_for_analysis = getattr(pl_instance, 'target_col', None)

    if target_col_for_analysis is None:
        st.error("Could not determine target column from the PseudoLabeler instance for analysis.")
    elif target_col_for_analysis not in combined_data.columns:
            st.error(f"Target column '{target_col_for_analysis}' not found in the combined data for distribution analysis.")
    else:
        st.subheader("1. Pseudo-Label Distribution")
        st.write(f"Distribution of the target variable '{target_col_for_analysis}' after pseudo-labeling.")
        
        # Use columns for side-by-side display
        col1_chart, col2_table = st.columns(2)
        with col1_chart:
            fig, ax = plt.subplots(figsize=(6, 4)) # Smaller chart size
            sns.countplot(x=combined_data[target_col_for_analysis], order=combined_data[target_col_for_analysis].value_counts().index, palette='viridis', ax=ax)
            ax.set_title(f"Distribution of {target_col_for_analysis}")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        with col2_table:
            st.write("Frequency Table:")
            st.dataframe(combined_data[target_col_for_analysis].value_counts().reset_index().rename(columns={'index': 'Label', target_col_for_analysis: 'Count'}))


        st.subheader("2. Confidence Distribution Histogram")
        st.write("This histogram shows the distribution of confidence scores for the pseudo-labeled data.")
        # Accessing 'last_confidences' from the imported PseudoLabeler instance
        if hasattr(pl_instance, 'last_confidences') and pl_instance.last_confidences is not None and pl_instance.last_confidences.size > 0:
            fig, ax = plt.subplots(figsize=(7, 5)) # Adjusted size
            sns.histplot(pl_instance.last_confidences, bins=20, kde=True, color='skyblue', edgecolor='black', ax=ax)
            ax.set_title("Confidence Distribution of Pseudo-Labels")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No confidence data available to display.")

        st.subheader("3. Custom Variable Analysis")
        st.write("Select variables and an analysis type to explore distributions or relationships.")

        available_cols = [col for col in combined_data.columns if col != target_col_for_analysis]
        
        selected_vars = st.multiselect(
            "Select variables for analysis",
            options=available_cols,
            key="analysis_vars_multiselect"
        )

        analysis_type = st.radio(
            "Choose analysis type",
            ("Univariate Analysis", "Crosstabulation with Target"),
            key="analysis_type_radio"
        )

        if selected_vars:
            if analysis_type == "Univariate Analysis":
                st.markdown("#### Univariate Distributions")
                for i in range(0, len(selected_vars), 1): # Process one variable at a time to allow flexible column layout
                    col_name = selected_vars[i]
                    st.markdown(f"**Distribution of '{col_name}'**")
                    
                    col_chart, col_table = st.columns(2) # Create columns for chart and table

                    if pd.api.types.is_numeric_dtype(combined_data[col_name]):
                        with col_chart:
                            fig, ax = plt.subplots(figsize=(6, 4)) # Smaller chart size
                            sns.histplot(combined_data[col_name].dropna(), bins=30, kde=True, color='lightgreen', edgecolor='black', ax=ax)
                            ax.set_title(f"Histogram of {col_name}")
                            ax.set_xlabel(col_name)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                        with col_table:
                            st.write("Descriptive Statistics:")
                            st.dataframe(combined_data[col_name].describe())
                    else: # Categorical
                        with col_chart:
                            fig, ax = plt.subplots(figsize=(6, 4)) # Smaller chart size
                            sns.countplot(x=combined_data[col_name], order=combined_data[col_name].value_counts().index, palette='pastel', ax=ax)
                            ax.set_title(f"Bar Chart of {col_name}")
                            ax.set_xlabel(col_name)
                            ax.set_ylabel("Count")
                            plt.xticks(rotation=45, ha='right')
                            st.pyplot(fig)
                        with col_table:
                            st.write("Frequency Table:")
                            st.dataframe(combined_data[col_name].value_counts().reset_index().rename(columns={'index': col_name, col_name: 'Count'}))
                    st.markdown("---") # Separator for clarity
            
            elif analysis_type == "Crosstabulation with Target":
                st.markdown("#### Crosstabulations with Target Variable")
                for i in range(0, len(selected_vars), 1): # Process one variable at a time for crosstab
                    col_name = selected_vars[i]
                    st.markdown(f"**Crosstabulation: '{col_name}' vs. '{target_col_for_analysis}'**")

                    col_table, col_chart = st.columns(2) # Create columns for table and chart

                    # Check if the selected variable is categorical
                    if pd.api.types.is_object_dtype(combined_data[col_name]) or str(combined_data[col_name].dtype) == 'category':
                        crosstab_table = pd.crosstab(combined_data[col_name], combined_data[target_col_for_analysis])
                        with col_table:
                            st.dataframe(crosstab_table)

                        with col_chart:
                            # Display as stacked bar chart for categorical vs. categorical using Seaborn
                            fig, ax = plt.subplots(figsize=(7, 5)) # Adjusted size
                            sns.countplot(data=combined_data, x=col_name, hue=target_col_for_analysis, palette='deep', ax=ax)
                            ax.set_title(f"Stacked Bar Chart of {col_name} by {target_col_for_analysis}")
                            ax.set_xlabel(col_name)
                            ax.set_ylabel("Count")
                            plt.xticks(rotation=45, ha='right')
                            plt.legend(title=target_col_for_analysis)
                            st.pyplot(fig)

                    # Check if the selected variable is numerical
                    elif pd.api.types.is_numeric_dtype(combined_data[col_name]):
                        with col_table:
                            st.write(f"Descriptive Statistics of '{col_name}' grouped by '{target_col_for_analysis}':")
                            st.dataframe(combined_data.groupby(target_col_for_analysis)[col_name].describe())

                        with col_chart:
                            # Display as box plot or violin plot for numerical vs. categorical target
                            fig, ax = plt.subplots(figsize=(7, 5)) # Adjusted size
                            sns.boxplot(x=target_col_for_analysis, y=col_name, data=combined_data, palette='coolwarm', ax=ax)
                            ax.set_title(f"Box Plot of {col_name} by {target_col_for_analysis}")
                            ax.set_xlabel(target_col_for_analysis)
                            ax.set_ylabel(col_name)
                            st.pyplot(fig)
                    else:
                        st.info(f"Analysis for '{col_name}' not supported for this data type or combination.")
                    st.markdown("---")
        else:
            st.info("Please select at least one variable to perform analysis.")
else:
    st.info("Please run the pseudo-labeling process in the 'Pseudo-Labeling' tab first to see analysis.")