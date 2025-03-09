import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---- FUNCTIONS ----

# Author : https://www.kaggle.com/gemartin/load-data-reduce-memory-usage and adapted to be able to check if the column is actually a float
#or is an error like a code being a float column - if the column has Nan it will be still a float and will give us a warning.

def reduce_memory_usage(df):
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
       Uses 32-bit types as the minimum size for optimization."""
    df = df.copy()
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype

        # Skip categorical columns
        if pd.api.types.is_categorical_dtype(col_type):
            continue

        if np.issubdtype(col_type, np.integer):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        
        elif np.issubdtype(col_type, np.floating):
            c_min = df[col].min()
            c_max = df[col].max()
            # Check if the column contains NaN values
            if not df[col].isna().any():
                # Check if all values are integers
                if (df[col] % 1 == 0).all():
                    # Convert to nullable integer type if compatible
                    if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    # Optimize as float32 if range allows
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        print(f"Column '{col}' converted to float32")
                    else:
                        print(f"Column '{col}' remains as float64 due to range")
            else:
                # Optimize as float32 if range allows
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    print(f"Column '{col}' with NaNs converted to float32")
                else:
                    print(f"Column '{col}' with NaNs remains as float64 due to range")
        
        elif col_type == 'object':
            # Convert to category if unique values are below a threshold
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

def encoder_and_scaler(scaler, df, numerical_columns):
    """
    Preprocess data by scaling numerical columns using the specified scaler.

    Args:
        scaler (str): "standard", "minmax", or "robust" - specifies the scaler to use.
        numerical_columns (list): List of numerical columns for scaling.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    # Define available scalers
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()}

    if scaler not in scalers:
        raise ValueError("Invalid scaler specified. Choose from 'standard', 'minmax', or 'robust'.")

    scaler_instance = scalers[scaler]

    # Process numerical columns
    scaled_numerical = None
    if numerical_columns:
        scaled_numerical = scaler_instance.fit_transform(df[numerical_columns])

    # Combine transformed data into a DataFrame
    transformed_dfs = []
    if scaled_numerical is not None:
        transformed_dfs.append(pd.DataFrame(scaled_numerical, columns=numerical_columns, index=df.index))

    # Include any columns not transformed
    other_columns = df.drop(columns=numerical_columns, errors='ignore')
    if not other_columns.empty:
        transformed_dfs.append(other_columns)

    # Concatenate all parts into a single DataFrame
    transformed_df = pd.concat(transformed_dfs, axis=1)

    return transformed_df


# ---- PREPARATION PIP ----

rename_dict = {
    'DOW_0': 'Sun_Orders',
    'DOW_1': 'Mon_Orders',
    'DOW_2': 'Tue_Orders',
    'DOW_3': 'Wed_Orders',
    'DOW_4': 'Thu_Orders',
    'DOW_5': 'Fri_Orders',
    'DOW_6': 'Sat_Orders'
}

Asian_Cuisine = [
    'CUI_Asian_spending_dist', 
    'CUI_Street Food / Snacks_spending_dist',
    'CUI_Japanese_spending_dist', 
    'CUI_Desserts_spending_dist',
    'CUI_Beverages_spending_dist',
    'CUI_Healthy_spending_dist'
]

Chinese_Cuisine = [
    'CUI_Chinese_spending_dist', 
    'CUI_Chicken Dishes_spending_dist', 
    'CUI_Noodle Dishes_spending_dist'
]

Western_Cuisine = [
    'CUI_Italian_spending_dist', 
    'CUI_Cafe_spending_dist', 
    'CUI_American_spending_dist'
]

Other_Cuisine = [
    'CUI_OTHER_spending_dist', 
    'CUI_Thai_spending_dist', 
    'CUI_Indian_spending_dist'
]

spending_behavior = ['Asian_Cuisine', 'Chinese_Cuisine', 'Western_Cuisine', 'Other_Cuisine', 'cuisine_diversity']
timing_behavior = ['Weekdays', 'Sat_Orders', 'Sun_Orders', 'breakfast_orders', 'lunch_orders',
                   'afternoon_snack_orders', 'dinner_orders', 'late_night_orders', 'time_of_day_diversity']
customer_behavior = ['antiguity', 'R_recency_days', 'F_total_orders',
                     'avg_products_p_order', 'avg_spend_p_order', 'M_total_spend']

all_perspectives = spending_behavior + timing_behavior + customer_behavior

def prepare_data(df):
    # Step 1: Drop duplicates
    # df.drop_duplicates(inplace=True)

    # Step 2: Fill missing values for `first_order`
    df['first_order'] = df['first_order'].fillna(0)

    # Step 3: Handle `HR_0` null values
    dow_columns = df.filter(regex=r'^DOW_\d').columns.tolist()
    hr_columns_excl_hr0 = ['HR_' + str(i) for i in range(1, 24)]
    df.loc[df['HR_0'].isnull(), 'HR_0'] = (
        df[dow_columns].sum(axis=1) - df[hr_columns_excl_hr0].sum(axis=1)
    )

    # Step 4: Replace "-" in `last_promo` and `customer_region`
    df["last_promo"] = df["last_promo"].replace("-", "No Promo")
    most_frequent_region = df.loc[df["customer_region"] != "-", "customer_region"].mode()[0]
    df["customer_region"] = df["customer_region"].replace("-", most_frequent_region)

    # Step 5: KNN Imputation for `customer_age`
    scaler = StandardScaler()
    df['customer_age_scaled'] = scaler.fit_transform(df[['customer_age']])
    knn_imputer = KNNImputer(n_neighbors=5)
    df['customer_age_scaled'] = knn_imputer.fit_transform(df[['customer_age_scaled']])
    df['customer_age'] = scaler.inverse_transform(df[['customer_age_scaled']])
    df.drop(columns=['customer_age_scaled'], inplace=True)

    # Step 6: Filter rows with specific conditions and provide warnings
    invalid_rows = []

    for idx, row in df.iterrows():
        if row['product_count'] == 0:
            invalid_rows.append((idx, "Your record has a 'product_count' of 0, please confirm this line."))
        elif row['vendor_count'] == 0:
            invalid_rows.append((idx, "Your record has a 'vendor_count' of 0, please confirm this line."))
        elif row.filter(regex=r'^CUI_').sum() == 0:
            invalid_rows.append((idx, "Your record has a total cuisine spending of 0, please confirm this line."))
        elif row.filter(regex=r'^HR_\d').sum() == 0:
            invalid_rows.append((idx, "Your record has no orders in any hour, please confirm this line."))
        elif row[dow_columns].sum() == 0:
            invalid_rows.append((idx, "Your record has no orders on any day, please confirm this line."))
        elif row[dow_columns].sum() != row.filter(regex=r'^HR_\d').sum():
            invalid_rows.append((idx, "Your record's total orders in all hours and all days do not match, please confirm this line."))
        elif row['first_order'] > row['last_order']:
            invalid_rows.append((idx, "Your record's has first_order after last_order, please confirm this line."))

    # If there are invalid rows, stop the process and show warnings
    if invalid_rows:
        warning_message = "\n".join([f"Row {idx}: {message}" for idx, message in invalid_rows])
        raise ValueError(f"Invalid records detected:\n{warning_message}")

    # Step 7: Apply manual filtering
    def apply_manual_filtering(df):
        outliers_manual = (
            (df['CUI_OTHER'] <= 122) &
            (df['CUI_Italian'] <= 168) &
            (df['CUI_Japanese'] <= 130) &
            (df['CUI_Beverages'] <= 139) &
            (df["CUI_Street Food / Snacks"] <= 203) &
            (df['CUI_Chinese'] <= 100) &
            (df['CUI_American'] <= 121) &
            (df['CUI_Asian'] <= 252) &
            (df['CUI_Indian'] <= 109) &
            (df['CUI_Chicken Dishes'] <= 50) &
            (df["CUI_Thai"] <= 64) &
            (df['CUI_Noodle Dishes'] <= 70) &
            (df['CUI_Desserts'] <= 74) &
            (df['CUI_Healthy'] <= 80) &
            (df['CUI_Cafe'] <= 90) &
            (df["DOW_0"] <= 11) &
            (df["DOW_1"] <= 13) &
            (df["DOW_3"] <= 14) &
            (df["DOW_5"] <= 15) &
            (df["DOW_6"] <= 14) &
            (df["HR_0"] <= 9) &
            (df["HR_1"] <= 9) &
            (df["HR_2"] <= 10) &
            (df["HR_4"] <= 10) &
            (df["HR_5"] <= 5) &
            (df["HR_7"] <= 9) &
            (df["HR_8"] <= 10) &
            (df["HR_9"] <= 11) &
            (df["HR_10"] <= 17) &
            (df["HR_11"] <= 14) &
            (df["HR_12"] <= 14) &
            (df["HR_13"] <= 12) &
            (df["HR_14"] <= 11) &
            (df["HR_15"] <= 10) &
            (df["HR_16"] <= 13) &
            (df["HR_17"] <= 15) &
            (df["HR_18"] <= 15) &
            (df["HR_19"] <= 13) &
            (df["HR_20"] <= 10) &
            (df["HR_21"] <= 6) &
            (df["HR_22"] <= 7) &
            (df["HR_23"] <= 6) &
            (df['vendor_count'] <= 32) &
            (df['product_count'] <= 97) &
            (df['is_chain'] <= 40)
        )
        outliers = df[~outliers_manual]
        return df[outliers_manual], outliers

    df, outliers = apply_manual_filtering(df)
    
    def feature_engineering(df):
        # Step 8: Add derived features
        df.rename(columns=rename_dict, inplace=True)
        cuisine_columns = [col for col in df.columns if col.startswith('CUI_')]
        dow_columns = [col for col in df.columns if col.endswith('_Orders')]
        hour_columns = [col for col in df.columns if col.startswith('HR_')]

        df['antiguity'] = df['last_order'].max() - df['first_order']
        df['R_recency_days'] = df['last_order'].max() - df['last_order']
        df['F_total_orders'] = df[dow_columns].sum(axis=1)
        df['M_total_spend'] = df[cuisine_columns].sum(axis=1)
        df['avg_days_btw_orders'] = df['antiguity'] / df['F_total_orders'].replace(0, 1)
        df['avg_spend_p_order'] = df['M_total_spend'] / df['F_total_orders'].replace(0, 1)
        df['avg_products_p_order'] = df['product_count'] / df['F_total_orders'].replace(0, 1)
        df['cuisine_diversity'] = df[cuisine_columns].gt(0).sum(axis=1)
        df['chain_preference_ratio'] = df['is_chain'] / df['F_total_orders'].replace(0, 1)
        df[[f'{col}_spending_dist' for col in cuisine_columns]] = (
            df[cuisine_columns].div(df['M_total_spend'], axis=0).fillna(0).round(2)
        )
        df['promo_used'] = df['last_promo'].apply(lambda x: 1 if x != 'No Promo' else 0)
        df['weekday_avg_orders'] = df[['Mon_Orders', 'Tue_Orders', 'Wed_Orders', 'Thu_Orders', 'Fri_Orders']].mean(axis=1)
        df['weekend_avg_orders'] = df[['Sun_Orders', 'Sat_Orders']].mean(axis=1)
        df['breakfast_orders'] = df[['HR_5', 'HR_6', 'HR_7', 'HR_8', 'HR_9', 'HR_10']].sum(axis=1)
        df['lunch_orders'] = df[['HR_11', 'HR_12', 'HR_13', 'HR_14']].sum(axis=1)
        df['afternoon_snack_orders'] = df[['HR_15', 'HR_16', 'HR_17']].sum(axis=1)
        df['dinner_orders'] = df[['HR_18', 'HR_19', 'HR_20', 'HR_21']].sum(axis=1)
        df['late_night_orders'] = df[['HR_22', 'HR_23', 'HR_0', 'HR_1', 'HR_2', 'HR_3', 'HR_4']].sum(axis=1)
        df['day_of_week_diversity'] = df[dow_columns].gt(0).sum(axis=1)
        df['time_of_day_diversity'] = df[['breakfast_orders', 'lunch_orders', 'afternoon_snack_orders', 'dinner_orders', 'late_night_orders']].gt(0).sum(axis=1)
        df['vendor_loyalty'] = df['vendor_count'] / df['F_total_orders'].replace(0, 1)
        
        return df
    
    df = feature_engineering(df)
    outliers = feature_engineering(outliers)

    # Step 9: Final feature selection
    def apply_manual_filtering_final(df):
        outliers_manual = (
            (df['F_total_orders'] <= 50) &
            (df['avg_spend_p_order'] <= 100) &
            (df['breakfast_orders'] <= 25) &
            (df['lunch_orders'] <= 25) &
            (df['late_night_orders'] <= 20) &
            (df['product_count'] <= 70) &
            (df['weekend_avg_orders'] <= 10)
        )
        outliers = df[~outliers_manual]
        return df[outliers_manual], outliers

    final_df, outliers_final = apply_manual_filtering_final(df)
    outliers = pd.concat([outliers, outliers_final])

    if len(final_df) > 0:
        final_df['Asian_Cuisine'] = final_df[Asian_Cuisine].sum(axis=1)
        final_df['Chinese_Cuisine'] = final_df[Chinese_Cuisine].sum(axis=1)
        final_df['Western_Cuisine'] = final_df[Western_Cuisine].sum(axis=1)
        final_df['Other_Cuisine'] = final_df[Other_Cuisine].sum(axis=1)
        final_df['Weekdays'] = final_df[['Mon_Orders', 'Tue_Orders', 'Wed_Orders', 'Thu_Orders', 'Fri_Orders']].mean(axis=1)
        scaled_df = encoder_and_scaler("standard", final_df, all_perspectives)
        scaled_df = reduce_memory_usage(scaled_df)
        final_df = reduce_memory_usage(final_df)
    else:
        outliers['Asian_Cuisine'] = outliers[Asian_Cuisine].sum(axis=1)
        outliers['Chinese_Cuisine'] = outliers[Chinese_Cuisine].sum(axis=1)
        outliers['Western_Cuisine'] = outliers[Western_Cuisine].sum(axis=1)
        outliers['Other_Cuisine'] = outliers[Other_Cuisine].sum(axis=1)
        outliers['Weekdays'] = outliers[['Mon_Orders', 'Tue_Orders', 'Wed_Orders', 'Thu_Orders', 'Fri_Orders']].mean(axis=1)
        scaled_df = pd.DataFrame() 
        outliers = reduce_memory_usage(outliers)
    
    return scaled_df, final_df, outliers

def extract_column_info(df):
    """
    Extracts unique values and formats for each column in the dataset.
    Provides categorical options, numerical ranges, and date format hints.
    """
    column_info = {}
    for col in df.columns:
        # Check for categorical columns (object or small number of unique values)
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            unique_values = df[col].dropna().unique().tolist()
            if len(unique_values) > 0:
                column_info[col] = unique_values
            else:
                column_info[col] = []  # No unique values found
        # Check for numerical columns
        elif np.issubdtype(df[col].dtype, np.number):
            if not df[col].dropna().empty:
                column_info[col] = (df[col].min(), df[col].max())
            else:
                column_info[col] = (None, None)  # Handle columns with no numerical values
        # Check for datetime columns
        elif np.issubdtype(df[col].dtype, np.datetime64):
            column_info[col] = "Date format (YYYY-MM-DD)"
        # Fallback for other datatypes
        else:
            column_info[col] = "Unsupported data type"
    
    return column_info


#---- MAIN INPUT PIP ----

# ---- Page Title ----
st.title("🤖 Cluster Prediction")

# ---- Load Data ----
@st.cache_data
def load_data():
    """
    Load and preprocess datasets.
    """
    #--- ORIGINAL DATA / SYSTEM DATA TO EXTRACT COLUMN TYPES ------
    data_url = "https://raw.githubusercontent.com/CatarinaGN/DM_interface_design/refs/heads/main/data/DM2425_ABCDEats_DATASET.csv"
    df = pd.read_csv(data_url, sep=',')
    data_url_w_clusters = "https://raw.githubusercontent.com/CatarinaGN/DM_interface_design/refs/heads/main/data/DM2425_ABCDEats_DATASET_w_Clusters.csv"
    df_w_clusters = pd.read_csv(data_url_w_clusters, sep=',', index_col='customer_id')

    # Set customer_id as the index 
    df = df.drop_duplicates(subset='customer_id').set_index('customer_id')

    return df, df_w_clusters

df, df_w_clusters = load_data()

# ---- Extract Column Information ----
def extract_column_info(df):
    """
    Extracts unique values and formats for each column in the dataset.
    Provides categorical options, numerical ranges, and date format hints.
    """
    column_info = {}

    for col in df.columns:
        # Check for categorical columns (object or small number of unique values)
        if df[col].dtype == 'object': # or df[col].nunique() < 10
            unique_values = df[col].dropna().unique().tolist()
            if len(unique_values) > 0:
                column_info[col] = unique_values
            else:
                column_info[col] = []  # No unique values found
        # Check for numerical columns
        elif np.issubdtype(df[col].dtype, np.number):
        
            if col.startswith("HR_"): 
                column_info[col] = (0, 50)
            elif col.startswith("DOW_"):  
                column_info[col] = (0, 50)
            elif col.startswith("CUI_"):  
                column_info[col] = (0, 500)
            elif not df[col].dropna().empty:
                column_info[col] = (df[col].min(), df[col].max())
            else:
                column_info[col] = (None, None)  # Handle columns with no numerical values
        # Check for datetime columns
        elif np.issubdtype(df[col].dtype, np.datetime64):
            column_info[col] = "Date format (YYYY-MM-DD)"
        # Fallback for other datatypes
        else:
            column_info[col] = "Unsupported data type"

    return column_info

# ---- Main Application ----
def main():
    # Extract column information for inputs
    column_info = extract_column_info(df)
    #st.write("Loaded DataFrame:", df.head())
    #st.write("Column Info Extracted:", column_info)


    st.write("### Enter Customer Details")
    input_data = {}

    # Create inputs dynamically based on column types
    for col, info in column_info.items():
        if col not in df.columns:
            continue
        if isinstance(info, list):  # Categorical
            if len(info) > 0:
                input_data[col] = st.selectbox(
                    f"{col}",
                    ["Select"] + info,
                    help=f"Select a value for {col} from the dropdown."
                )
            else:
                st.warning(f"No options available for {col}.")
                input_data[col] = st.text_input(
                    f"{col} (Enter manually)",
                    help=f"Enter a value for {col} manually."
                )
        elif isinstance(info, tuple):  # Numerical
            min_val, max_val = info
            if min_val is not None and max_val is not None:
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(min_val),
                    step=1.0,
                    help=f"Enter a value between {min_val} and {max_val} for {col}."
                )
            else:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=0.0,
                    step=1.0,
                    help=f"Enter a numerical value for {col}."
                )
        elif info == "Date format (YYYY-MM-DD)":  # Date
            input_data[col] = st.date_input(
                f"{col}",
                help=f"Select a date in YYYY-MM-DD format for {col}."
            )
        else:  # Fallback for unsupported data types
            input_data[col] = st.text_input(
                f"{col} (Unsupported type)",
                help=f"Enter a value for {col}. This field has an unsupported data type."
            )

    # ---- Prediction Logic ----
    if st.button("Predict Outcome"):
        try:
            # Validate inputs
            for key, value in input_data.items():
                if isinstance(value, str) and (value == "" or value == "Select"):
                    st.error(f"Please provide a valid input for {key}.")
                    return

            # Convert user input to DataFrame
            input_df = pd.DataFrame([input_data])
            st.write("Your Input:", input_df)
            # Preprocess the input data
            # Assuming prepare_data is defined elsewhere
            scaled_input, final_input, outlier = prepare_data(input_df)
           
            if len(outlier) == 0:
                st.write("Data for Prediction (after transformation):", final_input.head())
                # Find the nearest cluster in the dataset
                # Assuming all_perspectives is defined elsewhere
                X_clusters = df_w_clusters[all_perspectives]
                y_clusters = df_w_clusters['final_labels']

                # Measure distance and predict
                cluster_distances = ((X_clusters - final_input[all_perspectives].values) ** 2).sum(axis=1)
                closest_cluster = y_clusters.loc[cluster_distances.idxmin()]
                st.success(f"The record belongs to Cluster: {closest_cluster}")
            else:
                # Handle outlier by reclassification
                st.warning("Your record is an outlier. Reclassification using Decision Tree in progress...")
                st.write("Data for Prediction (after transformation):", outlier[all_perspectives].head())
                # Prepare data for decision tree
                all_features = df_w_clusters[all_perspectives]
                all_labels = df_w_clusters['final_labels']

                X_train, X_test, y_train, y_test = train_test_split(
                    all_features, all_labels, test_size=0.3, random_state=5
                )

                # Train decision tree - Decision trees and ensemble methods do not require 
                # feature scaling to be performed as they are not sensitive to the the variance in the data.
                # https://towardsdatascience.com/do-decision-trees-need-feature-scaling-97809eaa60c6
                dt = DecisionTreeClassifier(random_state=5, max_depth=3)
                dt.fit(X_train, y_train)

                # Predict outlier cluster
                outlier['final_labels'] = dt.predict(outlier[all_perspectives])
                st.success(f"The record was reclassified to Cluster: {outlier['final_labels'][0]}")

                # Show model performance
                accuracy = dt.score(X_test, y_test) * 100
                st.info(f"The decision tree estimates {accuracy:.2f}% accuracy on customer classification.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---- Run Application ----
main()
