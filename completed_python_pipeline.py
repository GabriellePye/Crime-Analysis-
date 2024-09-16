import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from geopy.geocoders import Nominatim
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Levels above debug
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_pipeline.log', # Pipeline log file
    filemode='w' # Overwrite logs
)

# Initialize geocoder
geolocator = Nominatim(user_agent='myGeocoder')

# Constants
DATA_FOLDER = './Data/' # Paths to each dataset file and their types are defined.
STAGING_FOLDER = './staging'
PRIMARY_FOLDER = './primary'
POLICE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'Police Dataset 2021-2024')
REPORTING_FOLDER = './reporting'
CACHE_FILE = 'postcode_cache.json'

STAGED_FILES = {
    'staged_police_data': 'staged_police_data.csv',
    'staged_cbp_8322_authority': 'staged_CBP-8322-authority.csv',
    'staged_cbp_8322_constituency': 'staged_CBP-8322-constituency.csv',
    'staged_cbp_7293': 'staged_CBP-7293.csv',
    'staged_english_la_name_codes': 'staged_EnglishLaNameCodes.csv',
    'staged_house_prices': 'staged_house_prices.csv'
}

PRIMARY_FILES = {
    'staged_police_data': 'primary_police_data.csv',
    'staged_cbp_8322_authority': 'primary_CBP-8322-authority.csv',
    'staged_cbp_8322_constituency': 'primary_CBP-8322-constituency.csv',
    'staged_cbp_7293': 'primary_CBP-7293.csv',
    'staged_english_la_name_codes': 'primary_EnglishLaNameCodes.csv',
    'staged_house_prices': 'primary_house_prices.csv'
}

REPORTING_FILES = {
    'primary_police_data': 'reporting_police_data.csv',
    'primary_CBP-8322-authority': 'reporting_CBP-8322-authority.csv',
    'primary_CBP-8322-constituency': 'reporting_CBP-8322-constituency.csv',
    'primary_CBP-7293': 'reporting_CBP-7293.csv',
    'primary_EnglishLaNameCodes': 'reporting_EnglishLaNameCodes.csv',
    'primary_house_prices': 'reporting_house_prices.csv'
}


# Define dataset file paths and types
CBP_8322_AUTHORITY_FILE = os.path.join(DATA_FOLDER, 'CBP-8322-authority.xlsx')
CBP_8322_CONSTITUENCY_FILE = os.path.join(DATA_FOLDER, 'CBP-8322-constituency.xlsx')
CBP_7293_FILE = os.path.join(DATA_FOLDER, 'CBP-7293.xlsx')
ENGLISH_LA_NAME_CODES_FILE = os.path.join(DATA_FOLDER, 'EnglishLaNameCodes.csv')
HOUSE_PRICES_FILE = os.path.join(DATA_FOLDER, 'house_prices.csv')
POLICE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'Police Dataset 2021-2024')

# Define file types
FILE_TYPES = {
    'CBP-8322-authority': 'excel',
    'CBP-8322-constituency': 'excel',
    'CBP-7293': 'excel',
    'EnglishLaNameCodes': 'csv',
    'house_prices': 'csv'
}

def load_data(file_path: str, file_type: str) -> pd.DataFrame: # Loads individual datasets based on file typ
    """Load a dataset from a given file path and type (csv or excel)."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Only 'csv' and 'excel' are supported.")
        logging.info(f"Successfully loaded {file_type} file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_type} file {file_path}: {e}")
        return None

def load_police_data(police_folder_path: str) -> pd.DataFrame: # Concatenates multiple police datasets into one DataFrame
    """Load and concatenate police datasets."""
    df_list = []
    for root, dirs, files in os.walk(police_folder_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                df = load_data(file_path, 'csv')
                if df is not None:
                    df_list.append(df)
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logging.info("Successfully concatenated police datasets.")
        return combined_df
    else:
        logging.error("No valid police datasets found to concatenate.")
        return None

def load_all_datasets(): # Uses constants to load all datasets and handle the police data separately.
    """Load all datasets using predefined constants."""
    files_and_types = {
        'CBP-8322-authority': (CBP_8322_AUTHORITY_FILE, FILE_TYPES['CBP-8322-authority']),
        'CBP-8322-constituency': (CBP_8322_CONSTITUENCY_FILE, FILE_TYPES['CBP-8322-constituency']),
        'CBP-7293': (CBP_7293_FILE, FILE_TYPES['CBP-7293']),
        'EnglishLaNameCodes': (ENGLISH_LA_NAME_CODES_FILE, FILE_TYPES['EnglishLaNameCodes']),
        'house_prices': (HOUSE_PRICES_FILE, FILE_TYPES['house_prices'])
    }
    
    datasets = {}
    for name, (file_path, file_type) in files_and_types.items():
        df = load_data(file_path, file_type)
        if df is not None:
            datasets[name] = df
    
    # Load police data separately
    police_df = load_police_data(POLICE_DATA_FOLDER)
    if police_df is not None:
        datasets['police'] = police_df
    
    return datasets

# FUNCTIONS 

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize text columns by converting to lowercase and stripping whitespace."""
    
    # Normalize column names (lowercase and replace spaces with underscores)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Identify object (string) columns dynamically
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # If there are no string columns, log and skip processing
    if not string_columns:
        logging.info("No string columns found for standardization.")
        return df
    
    # Standardize string columns (convert to lowercase, strip whitespace)
    for col in string_columns:
        logging.info(f"Standardizing column: {col}")
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on all columns if duplicates exist."""
    try:
        # Check for duplicates
        if df.duplicated().any():
            # Drop duplicate rows across all columns
            df = df.drop_duplicates()
            logging.info("Duplicates removed successfully.")
        else:
            logging.info("No duplicates found; skipping removal.")
            
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
    
    return df

def handle_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns with date-like data into datetime and create month/year columns if applicable."""
    
    # Identify date columns: columns that contain 'date' in their name or have date-like values
    date_columns = [col for col in df.columns if 'date' in col.lower() or df[col].astype(str).str.contains(r'\d{2}/\d{2}/\d{4}', na=False).any()]
    
    # If no date columns are found, log and return the DataFrame unchanged
    if not date_columns:
        logging.info("No date-like columns found.")
        return df
    
    logging.info(f"Converting date columns: {date_columns}")
    
    for col in date_columns:
        # Convert column to datetime
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # Create 'month' and 'year' columns based on the datetime column
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_year'] = df[col].dt.year

    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely empty if any exist."""
    # Identify columns that are entirely empty
    empty_cols = df.columns[df.isna().all()].tolist()
    
    if empty_cols:
        logging.info(f"Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
    else:
        logging.info("No empty columns to drop.")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling numerical columns with median values if necessary."""
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Check if there are any numerical columns with missing values
    missing_data = df[numerical_cols].isna().any().any()
    
    if not missing_data:
        # If no missing values are found in numerical columns, skip processing
        logging.info("No missing values found in numerical columns. Skipping missing values handling.")
        return df
    
    # If there are missing values, handle them
    for col in numerical_cols:
        if df[col].isna().any():
            logging.info(f"Handling missing values for column: {col}")
            df[col] = df[col].fillna(df[col].median())
    
    return df

def remove_outliers(df: pd.DataFrame, lat_range: tuple = (50, 52), lon_range: tuple = (-2, 2)) -> pd.DataFrame:
    """Remove outliers based on latitude and longitude ranges if necessary."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Check if any latitude or longitude values are outside the specified ranges
        outlier_lat = df['latitude'].notna() & ~df['latitude'].between(lat_range[0], lat_range[1])
        outlier_lon = df['longitude'].notna() & ~df['longitude'].between(lon_range[0], lon_range[1])
        
        if outlier_lat.any() or outlier_lon.any():
            logging.info("Removing outliers based on latitude and longitude.")
            original_size = len(df)
            df = df[~(outlier_lat | outlier_lon)]
            new_size = len(df)
            logging.info(f"Removed {original_size - new_size} rows with out-of-range lat/long values.")
        else:
            logging.info("No outliers found in latitude/longitude columns. Skipping outlier removal.")
    else:
        logging.info("No latitude/longitude columns found. Skipping outlier removal.")
    
    return df

# STAGING

def create_staging_folder():
    """Create the staging folder if it does not exist."""
    if not os.path.exists(STAGING_FOLDER):
        os.makedirs(STAGING_FOLDER)
        logging.info(f"Created staging folder: {STAGING_FOLDER}")
    else:
        logging.info(f"Staging folder already exists: {STAGING_FOLDER}")

def load_data(file_path: str, file_type: str) -> pd.DataFrame:
    """Load a dataset from a given file path and type (csv or excel)."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Only 'csv' and 'excel' are supported.")
        logging.info(f"Successfully loaded {file_type} file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_type} file {file_path}: {e}")
        return None

def load_police_data(police_folder_path: str) -> pd.DataFrame:
    """Load and concatenate police datasets."""
    df_list = []
    for root, dirs, files in os.walk(police_folder_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                df = load_data(file_path, 'csv')
                if df is not None:
                    df_list.append(df)
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logging.info("Successfully concatenated police datasets.")
        return combined_df
    else:
        logging.error("No valid police datasets found to concatenate.")
        return None

def save_to_csv(df: pd.DataFrame, file_path: str):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {e}")

def staging():
    """Load datasets, concatenate police data, and save as staged files."""
    create_staging_folder()
    
    # Load police data
    police_df = load_police_data(POLICE_DATA_FOLDER)
    if police_df is not None:
        save_to_csv(police_df, os.path.join(STAGING_FOLDER, 'staged_police_data.csv'))

    # Load other datasets
    files_and_types = {
        'CBP-8322-authority': (CBP_8322_AUTHORITY_FILE, FILE_TYPES['CBP-8322-authority']),
        'CBP-8322-constituency': (CBP_8322_CONSTITUENCY_FILE, FILE_TYPES['CBP-8322-constituency']),
        'CBP-7293': (CBP_7293_FILE, FILE_TYPES['CBP-7293']),
        'EnglishLaNameCodes': (ENGLISH_LA_NAME_CODES_FILE, FILE_TYPES['EnglishLaNameCodes']),
        'house_prices': (HOUSE_PRICES_FILE, FILE_TYPES['house_prices'])
    }
    
    for name, (file_path, file_type) in files_and_types.items():
        df = load_data(file_path, file_type)
        if df is not None:
            save_to_csv(df, os.path.join(STAGING_FOLDER, f'staged_{name}.csv'))

# Run the staging function
if __name__ == "__main__":
    staging()

# PRIMARY

def create_primary_folder():
    """Create the primary folder if it does not exist."""
    if not os.path.exists(PRIMARY_FOLDER):
        os.makedirs(PRIMARY_FOLDER)
        logging.info(f"Created primary folder: {PRIMARY_FOLDER}")
    else:
        logging.info(f"Primary folder already exists: {PRIMARY_FOLDER}")

def load_staged_data():
    """Load all staged datasets."""
    datasets = {}
    for name, file_name in STAGED_FILES.items():
        file_path = os.path.join(STAGING_FOLDER, file_name)
        if os.path.exists(file_path):
            datasets[name] = pd.read_csv(file_path)
            logging.info(f"Loaded staged dataset: {file_name}")
        else:
            logging.warning(f"Staged file not found: {file_name}")
    return datasets

def process_datasets(datasets: dict) -> None:
    """Process datasets: standardize, handle dates, and perform other operations."""
    for name, df in datasets.items():
        if df is not None:
            # Apply standardization
            df = standardize_data(df)
            
            # Drop empty columns if they exist
            if df.isna().all().any():
                df = drop_empty_columns(df)
            
            # Handle missing values if there are numerical columns
            if df.select_dtypes(include=['number']).columns.size > 0:
                df = handle_missing_values(df)
            
            # Handle dates if date columns are present
            if any(col for col in df.columns if 'date' in col.lower()):
                df = handle_dates(df)
            
            # Remove outliers if latitude and longitude columns are present
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = remove_outliers(df, lat_range=(50, 52), lon_range=(-2, 2))
            
            # Remove duplicates if there are any
            if df.duplicated().any():
                df = remove_duplicates(df)
            
            # Save processed data to primary folder
            primary_file_name = PRIMARY_FILES.get(name)
            if primary_file_name:
                save_to_csv(df, os.path.join(PRIMARY_FOLDER, primary_file_name))

def primary():
    """Main function for primary processing stage."""
    create_primary_folder()
    staged_data = load_staged_data()
    process_datasets(staged_data)
    logging.info("Primary stage processing completed.")

# Run the primary function
if __name__ == "__main__":
    primary()

# REPORTING FUNCTIONS 

def load_cache():
    """Load cache from a file, handle JSON errors, and return an empty dict if invalid."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading cache file {CACHE_FILE}: {e}")
            return {}
    return {}

def save_cache(cache):
    """Save the cache to a file."""
    try:
        with open(CACHE_FILE, 'w') as file:
            json.dump(cache, file, indent=4)  # Pretty-print JSON for easier debugging
    except IOError as e:
        logging.error(f"Error saving cache file {CACHE_FILE}: {e}")

def get_postcode(lat, lon, cache):
    """Convert latitude and longitude to postcode using Geopy with caching."""
    key = f"{lat},{lon}"
    if key in cache:
        return cache[key]
    
    geolocator = Nominatim(user_agent='Geop_crime_data')
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        address = location.raw.get('address', {})
        postcode = address.get('postcode', 'postcode not found')
    except Exception as e:
        postcode = 'postcode not found'
        logging.error(f"Error getting postcode for lat {lat}, lon {lon}: {e}")
    
    cache[key] = postcode
    save_cache(cache)
    return postcode

def add_postcode_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a postcode column to the DataFrame based on latitude and longitude."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        cache = load_cache()
        df['postcode'] = df.apply(
            lambda row: get_postcode(row['latitude'], row['longitude'], cache)
            if pd.notna(row['latitude']) and pd.notna(row['longitude'])
            else 'postcode not found',
            axis=1
        )
        # Optionally drop the latitude and longitude columns if no longer needed
        # df.drop(['latitude', 'longitude'], axis=1, inplace=True)
    else:
        logging.info("No latitude and longitude columns found. Skipping postcode conversion.")
    
    return df

# REPORTING 

def create_reporting_folder():
    """Create the reporting folder if it does not exist."""
    if not os.path.exists(REPORTING_FOLDER):
        os.makedirs(REPORTING_FOLDER)
        logging.info(f"Created reporting folder: {REPORTING_FOLDER}")
    else:
        logging.info(f"Reporting folder already exists: {REPORTING_FOLDER}")

def save_to_csv(df: pd.DataFrame, file_path: str):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {e}")

def process_police_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the police dataset with specific transformations."""
    if 'Month' in df.columns:
        # Convert 'Month' to 'Date', drop original 'Month' and extract new columns
        df['Date'] = pd.to_datetime(df['Month'], format='%Y-%m', errors='coerce')
        df.drop(columns=['Month'], inplace=True)
        df['Month'] = df['Date'].dt.strftime('%B')  # 'January', 'February', etc.
        df['Year'] = df['Date'].dt.year
    
    # Drop unnecessary columns
    columns_to_drop = ['Last outcome category', 'Context', 'Falls within']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Assign IDs for specific crime types
    prefix = 'ID_'
    if 'Crime type' in df.columns and 'Crime ID' in df.columns:
        df.loc[
            (df['Crime type'] == 'Anti-social behaviour') & (df['Crime ID'].isna()),
            'Crime ID'] = [f'{prefix}{i:03d}' for i in range(1, (df['Crime type'] == 'Anti-social behaviour').sum() + 1)]
    
    return df

def add_postcode_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a postcode column to the DataFrame based on latitude and longitude."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        cache = load_cache()
        df['postcode'] = df.apply(
            lambda row: get_postcode(row['latitude'], row['longitude'], cache)
            if pd.notna(row['latitude']) and pd.notna(row['longitude'])
            else 'postcode not found',
            axis=1
        )
        # Optionally drop the latitude and longitude columns if no longer needed
        # df.drop(['latitude', 'longitude'], axis=1, inplace=True)
    else:
        logging.info("No latitude and longitude columns found. Skipping postcode conversion.")
    
    return df

def reporting():
    """Final reporting stage."""
    create_reporting_folder()
    
    # Load primary datasets
    primary_files = {
        'primary_police_data': 'primary_police_data.csv',
        'primary_cbp_8322_authority': 'primary_CBP-8322-authority.csv',
        'primary_cbp_8322_constituency': 'primary_CBP-8322-constituency.csv',
        'primary_cbp_7293': 'primary_CBP-7293.csv',
        'primary_english_la_name_codes': 'primary_EnglishLaNameCodes.csv',
        'primary_house_prices': 'primary_house_prices.csv'
    }
    
    for name, file_name in primary_files.items():
        file_path = os.path.join(PRIMARY_FOLDER, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            if name == 'primary_police_data':
                df = process_police_data(df)
                df = add_postcode_column(df)  # Add postcode column if applicable
            
            # Save the processed DataFrame to the reporting folder
            save_to_csv(df, os.path.join(REPORTING_FOLDER, file_name))
        else:
            logging.warning(f"Primary file not found: {file_name}")

# Run the reporting function
if __name__ == "__main__":
    reporting()

# To execute pipeline all at once 

def main(pipeline='all'):
    logging.info("Pipeline execution started")

    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            if pipeline in ['all', 'staging']:
                staging()
                logging.info("Staging execution completed successfully")
                
            if pipeline in ['all', 'primary']:
                primary()
                logging.info("Primary execution completed successfully")
                
            if pipeline in ['all', 'reporting']:
                reporting()
                logging.info("Reporting execution completed successfully")
                
            logging.info("Pipeline run complete")
        
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
