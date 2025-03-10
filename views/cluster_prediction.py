import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
#outliers
from scipy.stats.mstats import winsorize
#encoding
from sklearn.preprocessing import OrdinalEncoder
#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

# ---- FUNCTIONS ----

# Author : https://www.kaggle.com/gemartin/load-data-reduce-memory-usage and adapted to be able to check if the column is actually a float
#or is an error like a code being a float column - if the column has Nan it will be still a float and will give us a warning.

final_features = ['Age','OtherRevenue','PersonsNights','Total_Revenue','RevenuePerPersonNight','AvgOccupancy','ADR','AverageLeadTime']


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

columns_to_treat = ['AvgOccupancy','ADR','RevenuePerPersonNight',
                    'OtherRevenue','Total_Revenue','LodgingRevenue',
                    'SpendingPerBooking','PersonsNights','AverageLeadTime',
                    'RoomNights','TotalSpecialRequests','Age']

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

def one_hot_encode(df, columns):
    
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
    return df_encoded


def pipeline(df, df_w_clusters, outlier_criteria, days_order, lead_order, columns_to_treat, features_to_scale, winsorization=True, scaling_method='min-max'):

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
        elif row['LodgingRevenue'] == 0 and row['OtherRevenue'] == 0:
            invalid_rows.append((idx, "Your record's total revenue is 0, please confirm this line."))
        elif row['RoomNights'] > row['PersonsNights']:
            invalid_rows.append((idx, "PersonNights should be greater than or equal to RoomNights. Please check this line."))

    # If there are invalid rows, stop the process and show warnings
    if invalid_rows:
        warning_message = "\n".join([f"Row {idx}: {message}" for idx, message in invalid_rows])
        raise ValueError(f"Invalid records detected:\n{warning_message}")
    
    df_preparation = df.copy()
    
    # Step 1: Scale 'Age' using StandardScaler
    scaler = StandardScaler()
    df_w_clusters['Age_scaled'] = scaler.fit_transform(df_w_clusters[['Age']])
    df_preparation['Age_scaled'] = scaler.transform(df_preparation[['Age']])  # Transform df_preparation

    # Step 2: KNN Imputation on df_preparation
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(df_w_clusters[['Age_scaled']])  # Fit on df_w_clusters
    df_preparation['Age_scaled'] = knn_imputer.transform(df_preparation[['Age_scaled']])  # Transform only df_preparation

    # Step 3: Reverse scaling to get back 'Age'
    df_preparation['Age'] = scaler.inverse_transform(df_preparation[['Age_scaled']])

    # Step 4: Drop the intermediate scaled column
    df_preparation.drop(columns=['Age_scaled'], inplace=True)
    df_w_clusters.drop(columns=['Age_scaled'], inplace=True)

    #Feature engineering
    df_preparation['BookingFrequency'] = df_preparation['BookingsCheckedIn'] + df_preparation['BookingsCanceled'] + df_preparation['BookingsNoShowed']
    df_preparation['BookingSuccessRate'] = np.where(df_preparation['BookingFrequency'] == 0, 0,
                                                df_preparation['BookingsCheckedIn'] / df_preparation['BookingFrequency'])
    total_special_requests = df_preparation['SRHighFloor'] + df_preparation['SRLowFloor'] + df_preparation['SRAccessibleRoom'] + \
                         df_preparation['SRMediumFloor'] + df_preparation['SRBathtub'] + df_preparation['SRShower'] + \
                         df_preparation['SRCrib'] + df_preparation['SRKingSizeBed'] + df_preparation['SRTwinBed'] + \
                         df_preparation['SRNearElevator'] + df_preparation['SRAwayFromElevator'] + \
                         df_preparation['SRNoAlcoholInMiniBar'] + df_preparation['SRQuietRoom']
    df_preparation['TotalSpecialRequests'] = total_special_requests
    df_preparation['SRFloor'] = df_preparation['SRHighFloor'] + df_preparation['SRMediumFloor'] + df_preparation['SRLowFloor']
    df_preparation['SRBed'] = df_preparation['SRKingSizeBed'] + df_preparation['SRTwinBed'] + df_preparation['SRCrib']
    df_preparation['SRNoisePreference'] = df_preparation['SRNearElevator'] + df_preparation['SRAwayFromElevator'] + df_preparation['SRQuietRoom']
    df_preparation['SRBathroom'] = df_preparation['SRBathtub'] + df_preparation['SRShower']
    df_preparation['Total_Revenue'] = df_preparation['LodgingRevenue'] + df_preparation['OtherRevenue']
    df_preparation['SpendingPerBooking'] = df_preparation['Total_Revenue'] / df_preparation['BookingFrequency'] 
    df_preparation['RevenuePerPersonNight'] = np.where(df_preparation['PersonsNights'] == 0, 0, df_preparation['Total_Revenue'] / df_preparation['PersonsNights'])
    df_preparation['AvgOccupancy'] = np.where(df_preparation['RoomNights'] == 0, 0, df_preparation['PersonsNights'] / df_preparation['RoomNights'])
    df_preparation['ADR'] = np.where(df_preparation['RoomNights'] == 0, 0, df_preparation['Total_Revenue'] / df_preparation['RoomNights'])
    df_preparation['SRFloor'] = (df_preparation['SRFloor'] > 0).astype(int)
    df_preparation['SRNoisePreference'] = (df_preparation['SRNoisePreference'] > 0).astype(int)
    df_preparation['SRBed'] = (df_preparation['SRBed'] > 0).astype(int)
    
    def categorize_booking_frequency(value):
        if value == 1:
            return "New Customer"
        else:
            return "Returning Customer"

    df_preparation["BookingFrequency"] = df_preparation["BookingFrequency"].apply(categorize_booking_frequency)
    df_preparation["BookingFrequency"].value_counts(normalize=True)

    # Calculate percentiles for categories
    days_percentiles = df_w_clusters['DaysSinceCreation'].quantile([0.25, 0.50, 0.75]).tolist() 
    lead_percentiles = df_w_clusters['AverageLeadTime'].quantile([0.25, 0.50, 0.75]).tolist()
    days_bins = [0] + days_percentiles + [float('inf')] #[0, 382.0, 717.0, 1019.0, inf]
    lead_bins = [0] + lead_percentiles + [float('inf')] #[0, 18.0, 59.0, 133.0, inf]

    # Category labels
    days_labels = ['Newly Registered', 'Developing', 'Established', 'Longstanding']
    lead_labels = ['Last-minute', 'Moderate Planners', 'Advance Planners', 'Long-term Planners']

    # Categorization
    df_preparation['DaysSinceCreation_Category'] = pd.cut(df_preparation['DaysSinceCreation'], bins=days_bins, labels=days_labels, include_lowest=True)
    df_preparation['AverageLeadTime_Category'] = pd.cut(df_preparation['AverageLeadTime'], bins=lead_bins, labels=lead_labels, include_lowest=True)

    outliers = pd.DataFrame()
    df_final = pd.DataFrame()
    
    # Remove outliers
    df_final, manual_loss = remove_outliers_manual(df_preparation, outlier_criteria)
    
      # Initialize as an empty DataFrame

    if manual_loss == 100.00:
        outliers = df_preparation.copy()

    if len(df_final) > 0:
        # Winsorization if enabled
        if winsorization:
            df_final = apply_winsorization(df_final, columns_to_treat)

        # Encoding
        df_final['Origin'] = df_final['Nationality'].map(country_map)
        df_final['Origin'] = df_final['Origin'].replace(
            ['Africa', 'Oceania', 'Antarctica', 'Asia', 'South_America'], 'Others')
        df_to_encode = df_final.copy()

    else:
        # Encoding
        outliers['Origin'] = outliers['Nationality'].map(country_map)
        outliers['Origin'] = outliers['Origin'].replace(
            ['Africa', 'Oceania', 'Antarctica', 'Asia', 'South_America'], 'Others')
        df_to_encode = outliers.copy()

    df_to_encode['DistributionChannel'] = df_to_encode['DistributionChannel'].replace(
        ['Travel Agent/Operator', 'GDS Systems'], 'Agent/Operator & GDS Systems')
    df_to_encode['BookingFrequency_bin'] = df_to_encode['BookingFrequency'].map(
        {'New Customer': 0, 'Returning Customer': 1})
    
    # Ordinal Encoding
    encoder = OrdinalEncoder(categories=[days_order, lead_order])
    df_to_encode[['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']] = encoder.fit_transform(
        df_to_encode[['DaysSinceCreation_Category', 'AverageLeadTime_Category']])
    df_to_encode[['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']] = df_to_encode[
        ['DaysSinceCreation_Encoded', 'AverageLeadTime_Encoded']].astype(int)
    
    # One-hot Encoding
    encoded_df = one_hot_encode(df_to_encode, ['DistributionChannel', 'Origin'])
    
    # Scaling
    scaled_df = scale_data(encoded_df, features_to_scale, method=scaling_method)
    
    return df_final, outliers, scaled_df


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

    data_url = r"data/Case1_HotelCustomerSegmentation.csv"  # Use raw string or forward slashes
    df_original = pd.read_csv(data_url, sep=';', index_col='ID')  # Corrected index assignment
    df_original.drop(columns=['NameHash', 'DocIDHash', 'MarketSegment'], inplace=True)    
    data_url_w_clusters = r"data/unscaled_df_umap.csv"
    df_w_clusters = pd.read_csv(data_url_w_clusters, sep=',')
    df_w_clusters.drop(columns=['MarketSegment'], inplace=True) 
    # Set customer_id as the index 
    #df_original = df_original.set_index('ID', inplace = True)

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
        
            if col.startswith("SR"): 
                column_info[col] = (0, 1)
            elif col.startswith("Age"):
                column_info[col] = (0, df[col].max())
            elif col.startswith("AverageLeadTime"):
                column_info[col] = (0, df[col].max())
            elif col.startswith("DaysSinceCreation"):
                column_info[col] = (0, df[col].max())
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

    #st.write("Loaded DataFrame:", df_original.head())
    column_info = extract_column_info(df_original)
    
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
            final_input, outlier, scaled_input = pipeline(input_df, df_w_clusters, outlier_criteria, days_order, lead_order, columns_to_treat, final_features, winsorization=True, scaling_method='min-max')

            if len(outlier) == 0:
                st.write("Data for Prediction (after transformation):", final_input.head())
                # Find the nearest cluster in the dataset
                X_clusters = df_w_clusters[final_features]
                y_clusters = df_w_clusters['KmLabels']

                # Measure distance and predict
                cluster_distances = ((X_clusters - final_input[final_features].values) ** 2).sum(axis=1)
                closest_cluster = y_clusters.loc[cluster_distances.idxmin()]
                st.success(f"The record belongs to Cluster: {closest_cluster}")
            else:
                # Handle outlier by reclassification
                st.warning("Your record is an outlier. Reclassification using Decision Tree in progress...")
                st.write("Data for Prediction (after transformation):", outlier[final_features].head())
                # Prepare data for decision tree
                all_features = df_w_clusters[final_features]
                all_labels = df_w_clusters['KmLabels']

                X_train, X_test, y_train, y_test = train_test_split(
                    all_features, all_labels, test_size=0.3, random_state=5
                )

                # Train decision tree - Decision trees and ensemble methods do not require 
                # feature scaling to be performed as they are not sensitive to the the variance in the data.
                # https://towardsdatascience.com/do-decision-trees-need-feature-scaling-97809eaa60c6
                dt = DecisionTreeClassifier(random_state=5, max_depth=3)
                dt.fit(X_train, y_train)

                # Predict outlier cluster
                outlier['final_labels'] = dt.predict(outlier[final_features])
                st.success(f"The record was reclassified to Cluster: {outlier['final_labels'][0]}")

                # Show model performance
                accuracy = dt.score(X_test, y_test) * 100
                st.info(f"The decision tree estimates {accuracy:.2f}% accuracy on customer classification.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---- Run Application ----
main()
