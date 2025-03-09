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
    """ 
    Optimize data types to reduce memory usage:
    - Converts integers to the smallest possible int type.
    - Converts floats to the smallest possible float type (at least float32).
    - Converts object columns to category if unique values are below a threshold.
    ✅ Evita float16 para prevenir erros de overflow/underflow.
    ✅ Mantém os float como float32 pelo menos, garantindo compatibilidade com cálculos estatísticos.
    """
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # Skip categorical columns
        if isinstance(col_type, pd.CategoricalDtype):
            continue

        # Optimize integer columns
        if np.issubdtype(col_type, np.integer):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Optimize float columns (avoid float16 to prevent precision issues)
        elif np.issubdtype(col_type, np.floating):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

        # Convert object columns to category if unique values are below a threshold
        elif df[col].dtype == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Convert if unique values < 50% of total rows
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Optimized memory usage: {end_mem:.2f} MB")
    print(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df



def scale_data(df, columns, method="min-max"):
    """
    Available methods: 'min-max', 'standardization', 'log', 'power'.
    """

    numeric_df = df.copy()
    
    if method == "log":
        numeric_df[columns] = np.log1p(numeric_df[columns])
    else:
        scaler = None
        if method == "min-max":
            scaler = MinMaxScaler()
        elif method == "standardization":
            scaler = StandardScaler()
        elif method == "power":
            scaler = PowerTransformer(method="yeo-johnson")
        else:
            raise ValueError("Invalid scaling method. Choose from 'min-max', 'standardization', 'log', 'power'.")
        
        numeric_df[columns] = scaler.fit_transform(numeric_df[columns]) if scaler else numeric_df[columns]

    return numeric_df


# ---- PREPARATION PIP ----

outlier_criteria = {"AvgOccupancy": (1, 6), 
                    "ADR": (0, 1400),
                    "RevenuePerPersonNight": (0, 2000), 
                    "Total_Revenue": (0, 20000),
                    "LodgingRevenue": (0, 20000), 
                    "Total_Revenue": (0, 8000),
                    "SpendingPerBooking": (0, 16000), 
                    "PersonsNights": (0, 175), 
                    "RoomNights": (0, 175)}

def remove_outliers_manual(df, conditions):
    df_filtered = df.copy()
    initial_rows = len(df_filtered)

    for column, (lower, upper) in conditions.items():
        df_filtered = df_filtered[(df_filtered[column] >= lower) & (df_filtered[column] <= upper)]

    data_loss = (1 - len(df_filtered) / initial_rows) * 100
    return df_filtered, round(data_loss, 2)


def apply_winsorization(df, numeric_columns, limits=(0, 0.025)):
    df_winsorized = df.copy()
    
    for col in numeric_columns:
        df_winsorized[col] = winsorize(df_winsorized[col], limits=limits)
    
    return df_winsorized

country_map = {
    'BRA': 'South_America','CHE': 'Other_Europe','FRA': 'France','DEU': 'Germany', 'CZE': 'Other_Europe',
    'AUT': 'Other_Europe','JPN': 'Asia','BEL': 'Other_Europe','GBR': 'United Kingdom','USA': 'North_America',
    'ESP': 'Other_Europe','PRT': 'Portugal','POL': 'Other_Europe','CHL': 'South_America','HUN': 'Other_Europe',
    'ISR': 'Asia','NLD': 'Other_Europe','MEX': 'North_America','SWE': 'Other_Europe','UKR': 'Other_Europe',
    'ARG': 'South_America','ITA': 'Other_Europe','GEO': 'Other_Europe','IRL': 'Other_Europe','ECU': 'South_America',
    'NOR': 'Other_Europe','PAN': 'North_America','DNK': 'Other_Europe','LUX': 'Other_Europe','RUS': 'Other_Europe',
    'CAN': 'North_America','ZAF': 'Africa','SVN': 'Other_Europe','FIN': 'Other_Europe','IND': 'Asia',
    'UZB': 'Asia','COD': 'Africa','JOR': 'Asia','ROU': 'Other_Europe','MAR': 'Africa','GRC': 'Other_Europe',
    'MUS': 'Africa','TUR': 'Asia','CHN': 'Asia','AUS': 'Oceania','SRB': 'Other_Europe','MLT': 'Other_Europe',
    'SGP': 'Asia','LVA': 'Other_Europe','ISL': 'Other_Europe','EST': 'Other_Europe','VEN': 'South_America',
    'MWI': 'Africa','IRN': 'Asia','BLR': 'Other_Europe','IDN': 'Asia','KOR': 'Asia','DOM': 'North_America',
    'CMR': 'Africa','SYR': 'Asia','SVK': 'Other_Europe','COL': 'South_America','PHL': 'Asia','MLI': 'Africa',
    'MOZ': 'Africa','PER': 'South_America','MYS': 'Asia','HRV': 'Other_Europe','CYP': 'Other_Europe','AGO': 'Africa',
    'URY': 'South_America','BGD': 'Asia','NZL': 'Oceania','BIH': 'Other_Europe','BGR': 'Other_Europe','ARM': 'Other_Europe',
    'THA': 'Asia','DZA': 'Africa','CRI': 'North_America','SAU': 'Asia','ATA': 'Antarctica', 'NGA': 'Africa',
    'AFG': 'Asia','VNM': 'Asia','CAF': 'Africa','LTU': 'Other_Europe','CPV': 'Africa','AZE': 'Other_Europe','KAZ': 'Asia',
    'GAB': 'Africa','SMR': 'Other_Europe','LBN': 'Asia','TWN': 'Asia','EGY': 'Africa','TGO': 'Africa','BOL': 'South_America',
    'GTM': 'North_America','JAM': 'North_America','PAK': 'Asia','TUN': 'Africa','UGA': 'Africa','ARE': 'Asia',
    'QAT': 'Asia','PRI': 'North_America','BRB': 'North_America','GHA': 'Africa','SEN': 'Africa','SLV': 'North_America',
    'IRQ': 'Asia','BEN': 'Africa','GIB': 'Other_Europe','LIE': 'Other_Europe','MDV': 'Asia','OMN': 'Asia',
    'ERI': 'Africa','CUB': 'North_America','SDN': 'Africa','MMR': 'Asia','MCO': 'Other_Europe','LAO': 'Asia',
    'MKD': 'Other_Europe','ALB': 'Other_Europe','ATF': 'Antarctica','ASM': 'Oceania','ATG': 'North_America','KGZ': 'Asia',
    'RWA': 'Africa','NAM': 'Africa','CIV': 'Africa','LCA': 'North_America','PRY': 'South_America','KIR': 'Oceania',
    'WLF': 'Oceania','LBY': 'Africa','KEN': 'Africa','GUY': 'South_America','KWT': 'Asia','JEY': 'Other_Europe',
    'BHR': 'Asia','SUR': 'South_America','CYM': 'North_America','HKG': 'Asia','YEM': 'Asia','COM': 'Africa',
    'BWA': 'Africa','LKA': 'Asia','FRO': 'Other_Europe','MDG': 'Africa','TJK': 'Asia','AIA': 'North_America',
    'SYC': 'Africa','TCD': 'Africa','SPM': 'North_America','SLE': 'Africa','SJM': 'Other_Europe','BMU': 'North_America',
    'SOM': 'Africa','TZA': 'Africa','GRD': 'North_America','TKM': 'Asia','NIC': 'North_America','ABW': 'North_America',
    'MNE': 'Other_Europe','STP': 'Africa','HTI': 'North_America','NCL': 'Oceania','GNB': 'Africa','PCN': 'Oceania',
    'VIR': 'North_America','AND': 'Other_Europe','GUF': 'South_America','WSM': 'Oceania','SWZ': 'Africa',
    'FLK': 'South_America','ETH': 'Africa','ZWE': 'Africa','MRT': 'Africa','DMA': 'North_America','FSM': 'Oceania'
}

# correct category order
days_order = ['Newly Registered', 'Developing', 'Established', 'Longstanding']
lead_order = ['Last-minute', 'Moderate Planners', 'Advance Planners', 'Long-term Planners']



def pipeline(df, outlier_criteria, days_order, lead_order, columns_to_treat, features_to_scale, winsorization=True, scaling_method='standardization'):
    df_preparation = df.copy()

    invalid_rows = []

    for idx, row in df.iterrows():
        if row['AverageLeadTime'] < 0:
            invalid_rows.append((idx, "Your record has a 'AverageLeadTime' smaller than 0, please confirm this line."))
        elif row['Age'] < 17:
            invalid_rows.append((idx, "Your record has a 'Age' smaller than 18, please confirm this line. According to our policies we are not segmenting clients with age bellow 18."))
        elif row['Age'] > 100:
            invalid_rows.append((idx, "Your record has a 'Age' bigger than 100, please confirm this line. According to our policies we are not segmenting clients with age above 100."))
        elif row['BookingsCheckedIn'] == 0:
            invalid_rows.append((idx, "Your record has no Bookings Checked-In, please confirm this line. Otherwise it is not part of the segmentation, is a Potential Client."))
        elif row['RoomNights'] == 0:
            invalid_rows.append((idx, "Your record has no RoomNights, please confirm this line."))
        elif row['PersonsNights'] == 0:
            invalid_rows.append((idx, "Your record has no PersonsNights, please confirm this line."))
        elif row[dow_columns].sum() != row.filter(regex=r'^HR_\d').sum():
            invalid_rows.append((idx, "Your record's total orders in all hours and all days do not match, please confirm this line."))
        elif row['first_order'] > row['last_order']:
            invalid_rows.append((idx, "Your record's has first_order after last_order, please confirm this line."))

    # If there are invalid rows, stop the process and show warnings
    if invalid_rows:
        warning_message = "\n".join([f"Row {idx}: {message}" for idx, message in invalid_rows])
        raise ValueError(f"Invalid records detected:\n{warning_message}")
    
        # Step 1: Fit the scaler and transform
    scaler = StandardScaler()
    df_w_clusters['Age_scaled'] = scaler.fit_transform(df_w_clusters[['Age']])
    df['Age_scaled'] = scaler.transform(df[['Age']])

    # Step 2: Apply KNN Imputation on the scaled data
    knn_imputer = KNNImputer(n_neighbors=5)
    df_w_clusters['Age_scaled'] = knn_imputer.fit(df_w_clusters[['Age_scaled']])
    df['Age_scaled'] = knn_imputer.fit_transform(df[['Age_scaled']])

    # Step 3: Reverse the scaling to get the original age range
    df_w_clusters['Age'] = scaler.inverse_transform(df_w_clusters[['Age_scaled']])
    df['Age'] = scaler.inverse_transform(df[['Age_scaled']])

    # Optional: Drop the intermediate scaled column if no longer needed
    df_w_clusters.drop(columns=['Age_scaled'], inplace=True)
    df.drop(columns=['Age_scaled'], inplace=True)
    
    # Remove outliers
    df_preparation, manual_loss = remove_outliers_manual(df_preparation, outlier_criteria)
    
    # Winsorization if enabled
    if winsorization:
        df_preparation = apply_winsorization(df_preparation, columns_to_treat)

    # Encoding
    df_preparation['Origin'] = df_preparation['Nationality'].map(country_map)
    df_preparation['Origin'] = df_preparation['Origin'].replace(
        ['Africa', 'Oceania', 'Antarctica', 'Asia', 'South_America'], 'Others')
    df_preparation['DistributionChannel'] = df_preparation['DistributionChannel'].replace(
        ['Travel Agent/Operator', 'GDS Systems'], 'Agent/Operator & GDS Systems')
    df_preparation['BookingFrequency_bin'] = df_preparation['BookingFrequency'].map(
        {'New Customer': 0, 'Returning Customer': 1})
    
    # Ordinal Encoding
    encoder = OrdinalEncoder(categories=[days_order, lead_order])
    df_preparation[['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']] = encoder.fit_transform(
        df_preparation[['DaysSinceCreation_Category', 'AverageLeadTime_Category']])
    df_preparation[['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']] = df_preparation[
        ['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']].astype(int)
    
    # One-hot Encoding
    encoded_df = one_hot_encode(df_preparation, ['DistributionChannel', 'Origin'])
    
    # Scaling
    scaled_df = scale_data(encoded_df, features_to_scale, method=scaling_method)
    
    return scaled_df








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
    df_original = pd.read_csv(data_url, sep=',')
    data_url_w_clusters = "https://raw.githubusercontent.com/CatarinaGN/DM_interface_design/refs/heads/main/data/DM2425_ABCDEats_DATASET_w_Clusters.csv"
    df_w_clusters = pd.read_csv(data_url_w_clusters, sep=',', index_col='customer_id')

    # Set customer_id as the index 
    df_original = df_original.set_index('ID', inplace = True)

    return df_original, df_w_clusters

df_original, df_w_clusters = load_data()

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
    column_info = extract_column_info(df_original)
    #st.write("Loaded DataFrame:", df.head())
    #st.write("Column Info Extracted:", column_info)


    st.write("### Enter Customer Details")
    input_data = {}

    # Create inputs dynamically based on column types
    for col, info in column_info.items():
        if col not in df_original.columns:
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
