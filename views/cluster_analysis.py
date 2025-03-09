import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from utils import run_tsne, create_figure

st.title("📊 Cluster Analysis")
tab1, tab2, tab3 = st.tabs(['Description', 'Viewer', 'Your Analysis Notes'])

with tab1:
    st.title('Things You Can Do! 😎')
    
    # Section: Cluster Comparison
    st.header('Cluster Comparison (Viewer Tab)')
    st.markdown("""
    ### In this space, you can analyze the clusters by comparing values of metric or categorical features across them. 
    Select the feature set you want on the sidebar, and then choose the type of plot you want. Here's what you can explore:

    #### Metric Features:
    - **Histograms**: Compare up to 4 different features. You can alternate between features.
    - **Violin Plots**: Visualize up to 4 different features. You can alternate between features.
    - **Radar Plot**: Choose the features you want to display.
    - **Table**: View mean values per cluster for the selected features.
    - **T-SNE Data in 2D and 3D Space Representation**: 
      - T-SNE is a **non-linear dimensionality reduction** algorithm designed for **exploration and visualization** of high-dimensional data. 
      - It maps high-dimensional points to a low-dimensional space while preserving the relationships between points.
      - Original Paper: [Maaten, L. v. d., & Hinton, G. E. (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
      - Adapted T-SNE Code: [noualibechir](https://github.com/noualibechir)

    #### Categorical Features:
    - **Bar Plots**: Visualize distributions for all categorical features across clusters.
    """)

    # Section: Individual Cluster Analysis
    st.header('Individual Cluster Analysis (Viewer Tab)')
    st.markdown("""
    ### **RFM Analysis**
    RFM analysis categorizes customers based on their purchasing behavior:
    - **Recency (R)**: How recent was the last transaction?
    - **Frequency (F)**: How often do they shop?
    - **Monetary (M)**: How much do they spend?

    #### RFM Scoring:
    Customers receive scores from 1 to 4 for each dimension based on quartiles:
    - **1**: Lowest value (least desirable for F and M, most desirable for R).
    - **4**: Highest value (most desirable for F and M, least desirable for R).

    #### Segmentation Categories:
    Using the combined RFM scores, customers are segmented into the following categories:
    - **Champions**: Recent, frequent, high spenders.
    - **Loyal Customers**: Recent and frequent but not the most recent.
    - **Potential Loyalists**: Recent but infrequent high spenders.
    - **At Risk**: Not recent, infrequent, low spenders.
    - **Hibernating Customers**: Older, infrequent, medium spenders.
    - **Lost Customers**: Low recency, frequency, and spending.
    - **Unknown Segment**: Unforeseen cases.

    Reference:
    - Sant’Anna, Larissa de. “Segmentação de Clientes Com RFM Em Python.” Medium, 22 Nov. 2023. [Read More](https://medium.com/@larixgomex/segmenta%C3%A7%C3%A3o-de-clientes-com-rfm-em-python-3a97e534ffa1)

    ### **Market Basket Analysis (MBA) Per Cluster**
    The Market Basket Analysis aims to discover frequently associated items in a shopping cart. 
    In our case, we analyze frequently associated cuisine types. These rules are useful for:
    - Understanding customer preferences.
    - Matching associated items for product promotions and communications.

    References:
    - IUYasik. “Market Basket Analysis & Apriori Algorithm Using Zhang’s Metric.” Medium, 4 Oct. 2023. [Read More](https://medium.com/@iuyasik/market-basket-analysis-apriori-algorithm-using-zhangs-metric-708406fc5dfc)
    - Yannawut Kimnaruk. “What Are Market Basket Analysis and the Apriori Algorithm?” Medium, 18 Sept. 2022. [Read More](https://yannawut.medium.com/what-are-market-basket-analysis-and-the-apriori-algorithm-fe0e8e6e34d)
    """)

    st.header('Take Notes While Analyzing:')
    st.markdown("""
    - While visualizing, you can take notes and see them latter in the **Your Analysis Notes** tab.
    - Your notes will be saved when you press the 'Save Note' button, so you can revisit them when needed. """)


with tab2:
    # Load data
    data_url_w_clusters = "https://raw.githubusercontent.com/CatarinaGN/DM_interface_design/refs/heads/main/data/DM2425_ABCDEats_DATASET_w_Clusters.csv"
    df_w_clusters = pd.read_csv(data_url_w_clusters, sep=',', index_col='customer_id')

    spending_behavior = ['Asian_Cuisine', 'Chinese_Cuisine', 'Western_Cuisine', 'Other_Cuisine', 'cuisine_diversity']
    timing_behavior = [
        'Weekdays', 'Sat_Orders', 'Sun_Orders', 'breakfast_orders', 'lunch_orders',
        'afternoon_snack_orders', 'dinner_orders', 'late_night_orders', 'time_of_day_diversity'
    ]
    customer_behavior = [
        'antiguity', 'R_recency_days', 'F_total_orders',
        'avg_products_p_order', 'avg_spend_p_order', 'M_total_spend'
    ]

    all_perspectives = spending_behavior + timing_behavior + customer_behavior

    analysis_columns = ['vendor_loyalty', 'Other_Cuisine','M_total_spend','lunch_orders', 'R_recency_days', 'avg_products_p_order', 'Sun_Orders',
                        'Sat_Orders', 'Chinese_Cuisine', 'avg_spend_p_order','time_of_day_diversity', 'Asian_Cuisine', 'cuisine_diversity',
                        'breakfast_orders', 'Western_Cuisine', 'late_night_orders','afternoon_snack_orders', 'chain_preference_ratio', 'antiguity','F_total_orders', 'dinner_orders',
                        'customer_age', 'vendor_count', 'product_count', 'is_chain','first_order', 'last_order','CUI_American', 'CUI_Asian', 'CUI_Beverages', 'CUI_Cafe',
                        'CUI_Chicken Dishes', 'CUI_Chinese', 'CUI_Desserts', 'CUI_Healthy','CUI_Indian', 'CUI_Italian', 'CUI_Japanese', 'CUI_Noodle Dishes','CUI_OTHER', 
                        'CUI_Street Food / Snacks', 'CUI_Thai', 'Mon_Orders','Tue_Orders', 'Wed_Orders', 'Thu_Orders', 'Fri_Orders', 'HR_0', 'HR_1',
                        'HR_2', 'HR_3', 'HR_4', 'HR_5', 'HR_6', 'HR_7', 'HR_8', 'HR_9', 'HR_10','HR_11', 'HR_12', 'HR_13', 'HR_14', 'HR_15', 'HR_16', 'HR_17', 'HR_18',
                        'HR_19', 'HR_20', 'HR_21', 'HR_22', 'HR_23', 'avg_days_btw_orders','CUI_American_spending_dist', 'CUI_Asian_spending_dist',
                        'CUI_Beverages_spending_dist', 'CUI_Cafe_spending_dist','CUI_Chicken Dishes_spending_dist', 'CUI_Chinese_spending_dist',
                        'CUI_Desserts_spending_dist', 'CUI_Healthy_spending_dist','CUI_Indian_spending_dist', 'CUI_Italian_spending_dist','CUI_Japanese_spending_dist',
                        'CUI_Noodle Dishes_spending_dist','CUI_OTHER_spending_dist', 'CUI_Street Food / Snacks_spending_dist',
                        'CUI_Thai_spending_dist', 'weekday_avg_orders','weekend_avg_orders', 'day_of_week_diversity', 'final_labels']

    hours = ['HR_0', 'HR_1', 'HR_2', 'HR_3', 'HR_4', 'HR_5', 'HR_6','HR_7', 'HR_8', 'HR_9', 
             'HR_10', 'HR_11', 'HR_12', 'HR_13', 'HR_14','HR_15', 'HR_16', 'HR_17', 'HR_18', 
             'HR_19', 'HR_20', 'HR_21', 'HR_22','HR_23']
    cuisines = ['CUI_American', 'CUI_Asian', 'CUI_Beverages', 'CUI_Cafe','CUI_Chicken Dishes', 'CUI_Chinese', 
                'CUI_Desserts', 'CUI_Healthy','CUI_Indian', 'CUI_Italian', 'CUI_Japanese', 'CUI_Noodle Dishes',
                'CUI_OTHER', 'CUI_Street Food / Snacks', 'CUI_Thai']
    days = ['Sun_Orders', 'Mon_Orders', 'Tue_Orders', 'Wed_Orders', 'Thu_Orders', 'Fri_Orders','Sat_Orders']

    categorical = ['customer_region', 'last_promo', 'payment_method', 'promo_used']

    ## ----- Cluster Comparison -----
    ### ----- Histograms -----
    def plot_interactive_histograms(data, labels_name):
        data = data[analysis_columns]
        unique_clusters = data[labels_name].unique()
        variables = [col for col in data.columns if col != labels_name]

        st.sidebar.write("### Select up to 4 variables to display:")
        selected_vars = st.sidebar.multiselect("Choose variables", variables, default=variables[:4])

        if len(selected_vars) > 4:
            st.error("You can select up to 4 variables only.")
            return

        st.sidebar.write("### Select clusters to include:")
        selected_clusters = st.sidebar.multiselect(
            "Choose clusters", unique_clusters, default=unique_clusters
        )

        filtered_data = data[data[labels_name].isin(selected_clusters)]

        # Render each selected variable as a separate histogram
        for selected_var in selected_vars:
            st.subheader(f"Histogram: {selected_var}")
            fig = px.histogram(
                filtered_data,
                x=selected_var,
                color=labels_name,
                barmode='overlay',
                title=f"Distribution of {selected_var} by Cluster",
                labels={labels_name: "Cluster"},
                opacity=0.7,
            )
            fig.update_layout(
                xaxis_title=selected_var,
                yaxis_title="Count",
                legend_title="Cluster",
                template="plotly_white"
            )
            st.plotly_chart(fig)

    ### ----- Violins -----
    def plot_interactive_violin(data, labels_name):
        data = data[analysis_columns]
        variables = [col for col in data.columns if col != labels_name]

        st.sidebar.write("### Select up to 4 variables to display:")
        selected_vars = st.sidebar.multiselect("Choose variables", variables, default=variables[:4])

        if len(selected_vars) > 4:
            st.error("You can select up to 4 variables only.")
            return

        st.sidebar.write("### Select clusters to include:")
        selected_clusters = st.sidebar.multiselect(
            "Choose clusters", data[labels_name].unique(), default=data[labels_name].unique()
        )

        filtered_data = data[data[labels_name].isin(selected_clusters)]

        # Render each selected variable as a separate violin plot
        for selected_var in selected_vars:
            st.subheader(f"Violin Plot: {selected_var}")
            fig = px.violin(
                filtered_data,
                x=labels_name,
                y=selected_var,
                color=labels_name,
                box=True,
                points=False,
                title=f"Violin Plot of {selected_var} by Cluster",
                labels={labels_name: "Cluster", selected_var: "Value"},
                template="plotly_white"
            )
            st.plotly_chart(fig)


    ## ----- Radar chart -----
    def plot_interactive_radar(data, labels_name):
        data = data[analysis_columns]

        # Allow users to select features to include in the radar chart
        st.sidebar.write("### Select features to include (minimum 3):")
        variables = [col for col in data.columns if col != labels_name]
        selected_features = st.sidebar.multiselect(
            "Choose features", variables, default=variables[:3]
        )

        # Enforce minimum of 3 features
        if len(selected_features) < 3:
            st.error("Please select at least 3 features.")
            return

        unique_clusters = data[labels_name].unique()

        # Normalize selected data
        normalized_data = data.copy()
        for var in selected_features:
            normalized_data[var] = (data[var] - data[var].min()) / (data[var].max() - data[var].min())

        st.sidebar.write("### Select clusters to display:")
        selected_clusters = st.sidebar.multiselect(
            "Choose clusters", unique_clusters, default=unique_clusters
        )

        fig = go.Figure()

        for cluster in selected_clusters:
            cluster_data = normalized_data[normalized_data[labels_name] == cluster][selected_features].mean()
            fig.add_trace(go.Scatterpolar(
                r=cluster_data.values.tolist(),
                theta=selected_features,
                fill='toself',
                name=f'Cluster {cluster}'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="Cluster Comparison (Normalized)",
            template="plotly_white"
        )
        st.plotly_chart(fig)

    ### --- tables ---
    def display_mean_table(data, labels_name):
        data = data[analysis_columns]
        variables = [col for col in data.columns if col != labels_name]

        st.sidebar.write("### Select features to include in the table:")
        selected_vars = st.sidebar.multiselect("Choose variables", variables, default=variables[:1])

        if not selected_vars:
            st.warning("Please select at least one feature to display.")
            return

        selected_clusters = st.sidebar.multiselect(
            "Choose clusters to include", data[labels_name].unique(), default=data[labels_name].unique()
        )

        filtered_data = data[data[labels_name].isin(selected_clusters)]

        # Calculate mean per cluster for the selected features
        grouped_data = filtered_data.groupby(labels_name)[selected_vars].mean().reset_index()

        st.subheader("Mean Values Table")
        st.dataframe(grouped_data)

    ### -- Bar plot for categorical features ---

    def plot_interactive_bar(data, labels_name):
        data = data[categorical + [labels_name]]
        variables = [col for col in data.columns if col != labels_name]
        unique_clusters = data[labels_name].unique()

        st.sidebar.write("### Select up to 4 categorical variables to display:")
        selected_vars = st.sidebar.multiselect("Choose variables", variables, default=variables[:4])

        if len(selected_vars) > 4:
            st.error("You can select up to 4 variables only.")
            return

        st.sidebar.write("### Select clusters to include:")
        selected_clusters = st.sidebar.multiselect(
            "Choose clusters", unique_clusters, default=unique_clusters
        )

        filtered_data = data[data[labels_name].isin(selected_clusters)]

        # Create a bar plot for each selected variable
        for selected_var in selected_vars:
            st.subheader(f"Bar Plot: {selected_var}")

            # Ensure the selected variable is treated as a categorical (string) type
            filtered_data[selected_var] = filtered_data[selected_var].astype(str)

            # Group and count data
            grouped_data = (
                filtered_data.groupby([labels_name, selected_var])
                .size()
                .reset_index(name='count')
            )

            # Pivot data
            pivot_data = grouped_data.pivot(index=selected_var, columns=labels_name, values='count').fillna(0)
            pivot_data.index = pivot_data.index.astype(str)  # Ensure index is categorical

            # Create the bar chart
            fig = go.Figure()
            for cluster in selected_clusters:
                fig.add_trace(go.Bar(
                    x=pivot_data.index,
                    y=pivot_data[cluster],
                    name=f"Cluster {cluster}"
                ))

            fig.update_layout(
                barmode='group',
                title=f"Distribution of {selected_var} by Cluster",
                xaxis_title=selected_var,
                yaxis_title="Count",
                legend_title="Cluster",
                template="plotly_white"
            )

            st.plotly_chart(fig)

# -----APPLICATION--------

    # Sidebar and analysis
    st.header("🔍 Analysis Options")
    analysis = st.selectbox(
        "Choose your analysis:",
        ["Cluster Comparison", "Individual Cluster Analysis"]
    )

    # Note-taking section
    st.header("📝 Write Your Notes")
    if 'notes' not in st.session_state:
        st.session_state['notes'] = []

    note_input = st.text_area("Write your insights here:", placeholder="Enter your analysis notes...")
    if st.button("Save Note"):
        if note_input:
            st.session_state['notes'].append(note_input)
            st.success("Note saved!")
        else:
            st.warning("Note cannot be empty.")

    if analysis == "Cluster Comparison":
        feature_type = st.sidebar.selectbox(
            "Choose a feature type:",
            ["Metric Features", "Categorical Features"]
        )

        if feature_type == "Metric Features":
            metric_option = st.sidebar.selectbox(
                "Choose a plot type:",
                ["Histogram", "Violin Plot", "Radar Chart", "Table", "t-SNE Visualization"]
            )

            if metric_option == "Histogram":
                plot_interactive_histograms(df_w_clusters, 'final_labels')  # Plot Histogram
            elif metric_option == "Violin Plot":
                plot_interactive_violin(df_w_clusters, 'final_labels')  # Plot Violin
            elif metric_option == "Radar Chart":
                plot_interactive_radar(df_w_clusters, 'final_labels')  # Plot Radar
            elif metric_option == "Table":
                display_mean_table(df_w_clusters, 'final_labels')  # Plot Table
            elif metric_option == "t-SNE Visualization":
                x_for_tsne = df_w_clusters[all_perspectives]
                data_for_tsne = df_w_clusters.copy()
                data_for_tsne.reset_index(inplace=True)
                final_labels = data_for_tsne['final_labels']

                st.sidebar.title('TSNE Visualization Settings')
                dimension = st.sidebar.selectbox('Choose the dimension of embedded space:', [2, 3], index=0)

                # --- Scale the Data ---
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(x_for_tsne)

                # --- Run t-SNE with Spinner ---
                with st.spinner('⏳ Running t-SNE...'):
                    Y_fold_3d = run_tsne(
                        x=scaled_data,
                        y=final_labels,
                        dimension=dimension,
                        random_state=42,
                        perplexity=30
                    )

                if not Y_fold_3d.empty:
                    st.success("🎉 t-SNE Completed Successfully!")

                    # --- Create and Display Plot ---
                    fig = create_figure(dimension=dimension, Y_fold_3d=Y_fold_3d)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Failed to generate the plot.")
                else:
                    st.error("❌ No data to visualize. Check previous steps.")

        elif feature_type == "Categorical Features":
            metric_option = st.sidebar.selectbox(
                "Choose a plot type:",
                ["Bar Plot"]
            )

            # ----- Application -----

            if metric_option == "Bar Plot":
                plot_interactive_bar(df_w_clusters, 'final_labels')  # Plot bar plot  

    elif analysis == "Individual Cluster Analysis":
        # Handle Individual Cluster Analysis logic

        analysis_type = st.sidebar.selectbox("Choose Analysis Type:", ["RFM Analysis", "Market Basket Analysis"])
        selected_cluster = st.sidebar.selectbox("Select a Cluster:", sorted(df_w_clusters['final_labels'].unique()))

        # Filter data by selected cluster
        cluster_data = df_w_clusters[df_w_clusters['final_labels'] == selected_cluster]

        if analysis_type == "RFM Analysis":
            st.header(f"📊 RFM Analysis for Cluster {selected_cluster}")
            # RFM function to assign RFM scores based on quartiles
            def RFMScore(x, col):
                if x <= col.quantile(0.25):
                    return '1'
                elif x <= col.quantile(0.5):
                    return '2'
                elif x <= col.quantile(0.75):
                    return '3'
                else:
                    return '4'
            #Previous notebook 

            # Calculate RFM scores
            cluster_data['RScore'] = cluster_data['R_recency_days'].apply(RFMScore, col=cluster_data['R_recency_days'])
            cluster_data['FScore'] = cluster_data['F_total_orders'].apply(RFMScore, col=cluster_data['F_total_orders'])
            cluster_data['MScore'] = cluster_data['M_total_spend'].apply(RFMScore, col=cluster_data['M_total_spend'])
            cluster_data['RFMScore'] = cluster_data['RScore'] + cluster_data['FScore'] + cluster_data['MScore']

            # Segment customers
            def segment_customers(score):
                R, F, M = score[0], score[1], score[2]  # Unpack the RFM score components
                
                # Define segment names based on RFM values
                if R == '4' and F in ['1', '2'] and M in ['1', '2']:
                    return 'Lost Customers'  # Low recency, low frequency, low monetary
                elif R == '1' and F in ['3', '4'] and M in ['3', '4']:
                    return 'Champions'  # Recent, frequent, and high spend
                elif R == '1' and F in ['1', '2'] and M in ['3', '4']:
                    return 'Potential Loyalists'  # Recent but low frequency, high spend
                elif R == '2' and F in ['3', '4'] and M in ['3', '4']:
                    return 'Loyal Customers'  # Recent, frequent, and high spend but not the most recent
                elif R in ['2', '3'] and F in ['1', '2'] and M in ['1', '2']:
                    return 'At Risk'  # Not recent, infrequent, low spend
                elif R == '2' and F in ['1', '2'] and M in ['3', '4']:
                    return 'Potential Loyalists'  # Moderate recency, low frequency, high spend
                elif R in ['3', '4'] and F in ['1', '2']:
                    return 'Hibernating Customers'  # Old, infrequent, mid spend
                else:
                    return 'Unknown Segment'  # For any other unforeseen cases
                
            cluster_data['CustomerSegment'] = cluster_data['RFMScore'].apply(segment_customers)

            # Aggregate RFM statistics
            RFM_segments = cluster_data.groupby(['CustomerSegment']).agg(
                NrCustomers=('CustomerSegment', 'size'),
                avgRecency=('R_recency_days', 'mean'),
                avgFrequency=('F_total_orders', 'mean'),
                avgMonetary=('M_total_spend', 'mean')
            ).fillna(0)

            st.subheader("RFM Segments Overview")
            st.dataframe(RFM_segments)

            # Optional: Additional visualizations
            st.subheader("Customer Distribution by Region")
            region_segment_distribution = cluster_data.groupby(['customer_region', 'CustomerSegment']).size().unstack(fill_value=0)
            st.dataframe(region_segment_distribution)

            st.subheader("Age Distribution by Segment")
            age_distribution = cluster_data.groupby('CustomerSegment')['customer_age'].mean()
            st.bar_chart(age_distribution)

            st.subheader("Order Distribution by Day of the Week")
            order_days_segment = cluster_data.groupby('CustomerSegment')[['Sun_Orders', 'Mon_Orders', 'Tue_Orders', 
                                                                        'Wed_Orders', 'Thu_Orders', 'Fri_Orders', 
                                                                        'Sat_Orders']].mean()
            st.dataframe(order_days_segment)

        elif analysis_type == "Market Basket Analysis":
            st.sidebar.subheader("Market Basket Analysis Settings")
            confidence_threshold = st.sidebar.slider(
                "Select Confidence Threshold:", 
                min_value=0.10, max_value=1.0, value=0.40, step=0.05)
            
            st.header(f"🛒 Market Basket Analysis for Cluster {selected_cluster}")

            from mlxtend.frequent_patterns import apriori, association_rules

            # Step 1: Filter the data for cuisine diversity and select relevant columns
            mba_data = cluster_data[all_perspectives + cuisines + ['final_labels']]
            cuisine_data = mba_data[mba_data['cuisine_diversity'] > 1].copy()
            cuisine_columns = [cuisine for cuisine in cuisines if cuisine != 'CUI_OTHER']
            cuisine_data = cuisine_data[cuisine_columns + ['final_labels']]
            
            # Drop the 'final_labels' column for MBA
            cluster_data_mba = cuisine_data.drop(columns=['final_labels'])
            
            # Ensure the data is boolean (required for Apriori)
            cluster_data_bool = cluster_data_mba.astype(bool)
            
            # Apply Apriori algorithm
            frequent_itemsets = apriori(cluster_data_bool, min_support=0.01, use_colnames=True)
            
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.10)
            
            # Filter rules by lift > 1 and confidence > thereshold
            filtered_rules = rules[(rules['lift'] > 1) & (rules['confidence'] > confidence_threshold)]
            
            # Sort rules by confidence and support
            filtered_rules.sort_values(by=['confidence', 'support'], ascending=False, inplace=True)
            
            # Step 3: Display Results
            if not filtered_rules.empty:
                st.write("Top 5 Association Rules:")
                filtered_rules_display = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5)
                st.dataframe(filtered_rules_display)
            else:
                st.write("No significant association rules found for this cluster.")


with tab3:
    st.title("📝 Your Analysis Notes")
    if st.session_state['notes']:
        for i, note in enumerate(st.session_state['notes']):
            st.write(f"**Note {i + 1}:** {note}")
    else:
        st.info("No notes saved yet.")

