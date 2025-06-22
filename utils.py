import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from pipeline.preprocessing import change_activity_duration, change_activity_sampling

# Function to perform hyperparameter tuning using Grid Search
def grid_search_hyperparameter_tuning(model, param_grid, X_train, y_train, 
                                      cv=5, scoring='roc_auc_ovr', n_jobs=-1, verbose=1):

    try:
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)

        # Fit the model to the training data
        grid_search.fit(X_train, y_train)

        # Retrieve best parameters and best score
        best_params = grid_search.best_params_ if grid_search.best_params_ else None
        best_score = grid_search.best_score_ if grid_search.best_score_ else 0.0

        print("\nBest Parameters:", best_params)
        print(f"Best {scoring} Score: {best_score:.4f}")

        return grid_search, best_params, best_score

    except Exception as e:
        print(f"Grid search failed with error: {str(e)}")
        return None, None, None


# train_data, test_data = subject_based_split(prepared_data)



def extract_features(data, with_magnitude):
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

def load_model(model_path):

    """    Load a pre-trained model from the specified path.
    Parameters:
    model_path (str): The path to the model file.   
    Returns:
    model: The loaded model.
    """
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None 
    



def prepare_main_dataset(raw_dataset, CODE_TO_CLASS, duration, frequency):
    """
    Prepare dataset by filtering activities, extracting features, and organizing data.
    
    Parameters:
    raw_dataset: Raw dataset containing 'activity', 'data', and 'subject' columns
    CODE_TO_CLASS: Dictionary mapping activity codes to class labels
    duration: Duration parameter for activity processing
    frequency: Frequency parameter for activity sampling
    
    Returns:
    pd.DataFrame: Prepared dataset with features, class labels, subjects, and activity codes
    """
    
    # Filter indices based on activity codes
    filtered_indices = []
    for i in raw_dataset.index:
        activity_code = raw_dataset['activity'][i]
        if activity_code in CODE_TO_CLASS:
            filtered_indices.append(i)

    print(f"Total samples: {len(raw_dataset)}")
    print(f"Filtered samples: {len(filtered_indices)}")

    print("Processing data...")
    feature_list = []
    labels = []
    subjects = []
    activity_codes = []

    for i in filtered_indices:
        # Get data and preprocess
        data = raw_dataset['data'][i]
        data = change_activity_duration(data, duration)
        data = change_activity_sampling(data, frequency)
        
        # Extract features
        features = extract_features(data, True)
        
        # Get class label
        activity_code = raw_dataset['activity'][i]
        class_label = CODE_TO_CLASS[activity_code]
        
        # Store results
        feature_list.append(features)
        labels.append(class_label)
        subjects.append(raw_dataset['subject'][i])
        activity_codes.append(activity_code)

    # Combine into final dataset
    prepared_dataset = pd.concat(feature_list, ignore_index=True)
    prepared_dataset['class'] = labels
    prepared_dataset['subject'] = subjects
    prepared_dataset['activity_code'] = activity_codes

    print(f"Final dataset shape: {prepared_dataset.shape}")
    print("\nClass distribution:")
    print(prepared_dataset['class'].value_counts())
    
    return prepared_dataset