import streamlit as st
import pandas as pd
import os  # For checking file existence

# Page title
st.title("📋 Your Suggestions")

# Name selection
name = st.selectbox("Select your name:", ["", "Alice", "Bob", "Charlie", "Dana", "Eve"])

# Subject selection
subject = st.selectbox(
    "Select the subject of your suggestion:",
    ["", "Visualization Ideas", "Cluster Strategies Ideas", "Other Ideas"]
)

# Cluster selection (only for "Cluster Strategies Ideas")
cluster = None
if subject == "Cluster Strategies Ideas":
    cluster = st.selectbox(
        "Select the cluster:",
        ["", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]
    )

# Suggestion text box
suggestion = st.text_area("Write your suggestion here:")

# Submit Button
if st.button("Submit Suggestion"):
    # Validation
    if not name:
        st.error("Please select your name.")
    elif not subject:
        st.error("Please select the subject of your suggestion.")
    elif subject == "Cluster Strategies Ideas" and not cluster:
        st.error("Please select a cluster.")
    elif not suggestion.strip():
        st.error("Please write your suggestion.")
    else:
        # Process and save the suggestion
        st.success("Thank you for your suggestion! This topic will be discussed in the next meeting 😉")
        st.write("### Summary of Your Input:")
        st.write(f"- **Name:** {name}")
        st.write(f"- **Subject:** {subject}")
        if cluster:
            st.write(f"- **Cluster:** {cluster}")
        st.write(f"- **Suggestion:** {suggestion}")

        # Prepare data
        data = {
            "Name": [name],
            "Subject": [subject],
            "Cluster": [cluster] if cluster else [None],
            "Suggestion": [suggestion]
        }
        df = pd.DataFrame(data)

        # Check if the file already exists
        file_path = "suggestions.csv"
        if not os.path.isfile(file_path):
            # File doesn't exist, create it with a header
            df.to_csv(file_path, mode="w", index=False, header=True)
        else:
            # File exists, append without header
            df.to_csv(file_path, mode="a", index=False, header=False)
