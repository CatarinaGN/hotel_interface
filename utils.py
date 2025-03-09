import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- t-SNE Function with Debugging ---
def run_tsne(x, y, dimension, random_state, perplexity):
    try:
        model = TSNE(n_components=dimension, random_state=random_state, perplexity=perplexity)
        Y_fold_3d = model.fit_transform(x)
        Y_fold_3d = pd.DataFrame(Y_fold_3d, columns=[f'dim {i + 1}' for i in range(dimension)])
        Y_fold_3d["final_labels"] = y.astype(str)

        # Debugging: Check output
        #st.write("✅ t-SNE DataFrame Shape:", Y_fold_3d.shape)
        #st.write("✅ t-SNE DataFrame Preview:", Y_fold_3d.head())

        return Y_fold_3d

    except Exception as e:
        st.error(f"❌ t-SNE failed with error: {e}")
        return pd.DataFrame()

# --- Plotting Function with Debugging ---
def create_figure(dimension, Y_fold_3d):
    traces = []

    if Y_fold_3d.empty:
        st.error("❌ No data to plot. Check t-SNE output.")
        return None

    try:
        if dimension == 3:
            for index, group in Y_fold_3d.groupby('final_labels'):
                st.write(f"📊 3D Plot - Group {index}: {group.shape[0]} points")
                scatter = go.Scatter3d(
                    name=f'Class {index}',
                    x=group['dim 1'],
                    y=group['dim 2'],
                    z=group['dim 3'],
                    mode='markers',
                    marker=dict(size=4, symbol='circle')
                )
                traces.append(scatter)

            fig = go.Figure(traces)
            fig.update_layout(
                width=800, height=600,
                margin=dict(l=40, r=40, b=40, t=40),
                title="3D t-SNE Visualization"
            )

        else:  # 2D Plot
            for index, group in Y_fold_3d.groupby('final_labels'):
                st.write(f"📊 2D Plot - Group {index}: {group.shape[0]} points")
                scatter = go.Scatter(
                    name=f'Class {index}',
                    x=group['dim 1'],
                    y=group['dim 2'],
                    mode='markers',
                    marker=dict(size=4.5, symbol='circle')
                )
                traces.append(scatter)

            fig = go.Figure(traces)
            fig.update_layout(
                width=800, height=600,
                margin=dict(l=40, r=40, b=40, t=40),
                title="2D t-SNE Visualization"
            )

        return fig

    except Exception as e:
        st.error(f"❌ Plot creation failed: {e}")
        return None
