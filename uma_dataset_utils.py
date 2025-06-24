import itertools
import numpy as np
import pandas as pd

import os
import re

def uma_search_csv_files(directory, activities_of_interest=None):
    csv_files = []
    for current_folder, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if activities_of_interest is not None:
                    for activity in activities_of_interest:
                        # Use regex to match exact activity number
                        pattern = rf"{activity}(?=Trial|_|\.csv)"
                        if re.search(pattern, file):
                            full_path = os.path.join(current_folder, file)
                            csv_files.append(full_path)
                            break
                else:
                    full_path = os.path.join(current_folder, file)
                    csv_files.append(full_path)
    
    

    combined_uma_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    combined_uma_df.rename(columns={
        'Accelerometer: x-axis (g)': 'acc_x',
        'Accelerometer: y-axis (g)': 'acc_y',
        'Accelerometer: z-axis (g)': 'acc_z',
        'Gyroscope: x-axis (rad/s)': 'gyro_x',
        'Gyroscope: y-axis (rad/s)': 'gyro_y',
        'Gyroscope: z-axis (rad/s)': 'gyro_z'
    }, inplace=True)

    return combined_uma_df






def group_sensor_data_by_segments(combined_df, include_gyroscope=True):
    """
    Groups sensor data into segments based on subject, activity, trial, and timestamp breaks.
    
    Args:
        combined_df: DataFrame with columns ['Subject', 'Activity', 'Trial', 'TimeStamp', 
                    'acc_x', 'acc_y', 'acc_z'] and optionally ['gyro_x', 'gyro_y', 'gyro_z']
        include_gyroscope: Boolean flag to include gyroscope data in segments
    
    Returns:
        DataFrame with grouped sensor data segments
    """
    grouped_data = []

    # Sort by Subject, Activity, Trial, and TimeStamp to ensure proper ordering
    combined_df_sorted = combined_df.sort_values(['Subject', 'Activity', 'Trial', 'TimeStamp']).reset_index(drop=True)

    current_group = []
    current_subject = None
    current_activity = None
    current_trial = None
    group_id = 0

    def create_group_row(group_data, subject, activity, trial, group_id, include_gyro):
        """Helper function to create a group row dictionary."""
        group_df = pd.DataFrame(group_data)
        
        if include_gyro:
            sensor_data = group_df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        else:
            sensor_data = group_df[['acc_x', 'acc_y', 'acc_z']]
        
        timestamps = group_df['TimeStamp']
        
        return {
            'subject': subject,
            'activity': f"D{activity}",
            'trial': trial,
            'group_id': group_id,
            'data': sensor_data,
            'timestamps': timestamps,
            'start_time': timestamps.iloc[0],
            'end_time': timestamps.iloc[-1],
            'duration': len(timestamps),
            'sampling_rate': len(timestamps) / (timestamps.iloc[-1] - timestamps.iloc[0]) if len(timestamps) > 1 else 0
        }

    for idx, row in combined_df_sorted.iterrows():
        # Check if we're starting a new session (different subject/activity/trial)
        if (current_subject != row['Subject'] or 
            current_activity != row['Activity'] or 
            current_trial != row['Trial']):
            
            # Save the previous group if it exists
            if current_group:
                group_row = create_group_row(current_group, current_subject, current_activity, 
                                           current_trial, group_id, include_gyroscope)
                grouped_data.append(group_row)
                group_id += 1
            
            # Start new group
            current_group = [row.to_dict()]
            current_subject = row['Subject']
            current_activity = row['Activity']
            current_trial = row['Trial']
        else:
            # Check if timestamp has reset/jumped backwards (indicating new data segment)
            if current_group:
                last_timestamp = current_group[-1]['TimeStamp']
                current_timestamp = row['TimeStamp']
                
                # If current timestamp is less than or equal to previous (reset occurred)
                if current_timestamp <= last_timestamp:
                    # Save the current group
                    group_row = create_group_row(current_group, current_subject, current_activity, 
                                               current_trial, group_id, include_gyroscope)
                    grouped_data.append(group_row)
                    group_id += 1
                    
                    # Start new group with current row
                    current_group = [row.to_dict()]
                else:
                    # Continue adding to current group
                    current_group.append(row.to_dict())
            else:
                current_group.append(row.to_dict())

    # Don't forget the last group
    if current_group:
        group_row = create_group_row(current_group, current_subject, current_activity, 
                                   current_trial, group_id, include_gyroscope)
        grouped_data.append(group_row)

    # Convert to DataFrame
    dataset = pd.DataFrame(grouped_data)
    
    # Print statistics
    print(f"Created {len(dataset)} data segments based on timestamp breaks")
    print(f"Segments per activity:")
    activity_counts = dataset['activity'].value_counts()
    print(activity_counts)
    print(f"\nTimestamp information:")
    print(f"Average segment duration: {dataset['duration'].mean():.1f} samples")
    print(f"Average sampling rate: {dataset['sampling_rate'].mean():.1f} Hz")
    
    return dataset

# Usage:
# uma_dataset = group_sensor_data_by_segments(combined_uma_df, INCLUDE_GYROSCOPE)
def extract_uma_features(data, with_magnitude):
    """
    Extracts various features from the time and frequency domains from a given sample of activity. Also constructs
    features by combining the raw data.

    :param data: the data from the activity
    :param with_magnitude: calculate the magnitude of the sensors
    :return: list with all the features extracted from the activity
    """
    
    # Make a copy to avoid modifying original data
    data = data.copy()
    
    # Clean and ensure all data is numeric
    for col in data.columns:
        # Convert to string first, then replace any concatenated values
        data[col] = data[col].astype(str)
        # If values are concatenated, this won't work well, so let's handle it
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill any NaN values with 0
    data = data.fillna(0)
    
    # Check if we still have any non-numeric data
    if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
        print("Warning: Non-numeric data detected, attempting to clean...")
        # Additional cleaning if needed
        for col in data.columns:
            if data[col].dtype == 'object':
                print(f"Column {col} contains non-numeric data")
                data[col] = 0  # Set problematic columns to 0
    
    # Calculates the acceleration and rotation magnitudes
    if with_magnitude:
        for i in range(0, data.shape[1], 3):
            if i + 2 < data.shape[1]:  # Ensure we have 3 columns
                # Select the 3-axis slice
                slice_ = data.iloc[:, i:i+3]
                
                # Compute magnitude
                magnitude = np.linalg.norm(slice_, axis=1)
                
                # Generate magnitude column name
                name = 'mag_' + data.columns[i][0:len(data.columns[i])-2]
                
                # Assign magnitude column
                data[name] = magnitude

    # Creates features vector name
    names = ['mean', 'var', 'std', 'median', 'max', 'min', 'ptp', 'centile25', 'centile75']
    columns = list('_'.join(n) for n in itertools.product(names, data.columns.tolist()))

    # Time domain features
    features = np.mean(data, axis=0)
    features = np.hstack((features, np.var(data, axis=0)))
    features = np.hstack((features, np.std(data, axis=0)))
    features = np.hstack((features, np.median(data, axis=0)))
    features = np.hstack((features, np.max(data, axis=0)))
    features = np.hstack((features, np.min(data, axis=0)))
    features = np.hstack((features, np.ptp(np.asarray(data), axis=0)))
    features = np.hstack((features, np.percentile(data, 25, axis=0)))
    features = np.hstack((features, np.percentile(data, 75, axis=0)))

    # Creates a DataFrame
    features = pd.DataFrame([features], columns=columns)
    return features



def prepare_uma_dataset(uma_dataset, include_gyroscope=True):
    feature_list = []
    labels = []
    subjects = []
    activity_codes = []

    for i in uma_dataset.index:
        # Get data from each row
        data = uma_dataset['data'][i]
        
        # Extract features
        features_uma = extract_uma_features(data, True)
        
        # Store results
        feature_list.append(features_uma)
        subjects.append(uma_dataset['subject'][i])
        activity_codes.append(uma_dataset['activity'][i])

    # Combine into final dataset
    uma_prepared_dataset = pd.concat(feature_list, ignore_index=True)
    uma_prepared_dataset['subject'] = subjects
    uma_prepared_dataset['activity_code'] = activity_codes

    print(f"UMA dataset shape: {uma_prepared_dataset.shape}")
    print("\nActivity code distribution:")
    print(uma_prepared_dataset['activity_code'].value_counts())

    return uma_prepared_dataset
