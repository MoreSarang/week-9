import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A class that takes categorical data and continuous values,
    determines which group a new observation falls into, and predicts
    an estimate value based on the provided data.
    """
    
    def __init__(self, estimate):
        """
        Initialize the GroupEstimate class.
        
        Parameters
        ----------
        estimate : str
            The type of estimate to use. Must be either "mean" or "median".
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.columns_ = None
    
    def fit(self, X, y):
        """
        Fit the model by calculating group estimates.
        
        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame of categorical data.
        y : array-like
            1-D array of continuous values corresponding to X.
        
        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Store column names for later
        self.columns_ = list(X.columns)
        
        # Create a copy of X to avoid modifying original
        df_combined = X.copy()
        
        # Add y as a new column
        df_combined['_target'] = y
        
        # Group by all columns in X
        grouped = df_combined.groupby(self.columns_)
        
        # Calculate mean or median for each group
        if self.estimate == "mean":
            self.group_estimates_ = grouped['_target'].mean()
        else:  # median
            self.group_estimates_ = grouped['_target'].median()
        
        return self
    
    def predict(self, X_):
        """
        Predict estimates for new observations.
        
        Parameters
        ----------
        X_ : array-like or pandas.DataFrame
            Array or DataFrame of observations corresponding to the columns in X.
        
        Returns
        -------
        numpy.ndarray
            Array of predicted estimates. Returns NaN for missing groups.
        """
        if self.group_estimates_ is None:
            raise ValueError("Model must be fitted before calling predict")
        
        # Convert to DataFrame if necessary
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.columns_)
        
        # Initialize predictions array
        predictions = np.zeros(len(X_))
        missing_count = 0
        
        # For each row, look up the corresponding group estimate
        for i, row in X_.iterrows():
            # Create tuple for index lookup
            if len(self.columns_) == 1:
                key = row[self.columns_[0]]
            else:
                key = tuple(row[self.columns_])
            
            # Check if group exists in fitted data
            try:
                predictions[i] = self.group_estimates_.loc[key]
            except KeyError:
                predictions[i] = np.nan
                missing_count += 1
        
        # Print message if there are missing groups
        if missing_count > 0:
            print(f"Warning: {missing_count} observation(s) had missing groups")
        
        return predictions
