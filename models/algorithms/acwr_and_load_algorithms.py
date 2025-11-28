"""
Algorithms for Training Load Calculations
Purpose: Core algorithms for ACWR, monotony, strain, and spike detection
Author: Samuel Greeman (Sports Data Science)
Integrated by: Harsha Prakash
Date: November 2025
Project: Athlete Load Management & Performance Optimization Platform

Algorithm 1: ACWR Calculator
Algorithm 2: Training Monotony Calculator
Algorithm 3: Training Strain Calculator
Algorithm 4: Load Spike Detector
Algorithm 5: Load Aggregation

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Algorithm 1

class ACWRCalculator:
    """
    - 0.8 - 1.3 is good
    - Higher than 1.5 or lower than 0.8 is considered unsafe
    """
    
    def __init__(self, acute_window: int = 7, chronic_window: int = 28):
        self.acute_window = acute_window
        self.chronic_window = chronic_window
    
    def calculate(self, df: pd.DataFrame, method: str = 'rolling_average') -> pd.DataFrame:
        """
        Algorithm (Rolling Average Method):
        1. Sort data by athlete and date
        2. For each athlete:
           a. Calculate 7-day rolling mean (acute)
           b. Calculate 28-day rolling mean (chronic)
           c. Compute ratio: ACWR = acute / chronic
        3. Categorize risk level
        
        Specified inputs:
            method: 'rolling_average' or 'ewma'
            
        Desired output:
            DataFrame with acute_load, chronic_load, acwr columns added
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        if method == 'rolling_average':
            result['acute_load'] = (result.groupby('athlete_id')['session_load'].transform(lambda x: x.rolling(window=self.acute_window, min_periods=1).mean()))
            result['chronic_load'] = (result.groupby('athlete_id')['session_load'].transform(lambda x: x.rolling(window=self.chronic_window, min_periods=1).mean()))
        elif method == 'ewma':
            result['acute_load'] = (result.groupby('athlete_id')['session_load'].transform(lambda x: x.ewm(span=self.acute_window).mean()))
            result['chronic_load'] = (result.groupby('athlete_id')['session_load'].transform(lambda x: x.ewm(span=self.chronic_window).mean()))
        result['acwr'] = result['acute_load'] / result['chronic_load']
        result['acwr'] = result['acwr'].replace([np.inf, -np.inf], np.nan)
        result['acwr_category'] = result['acwr'].apply(self._categorize_acwr)
        return result
    
    @staticmethod
    def _categorize_acwr(acwr: float) -> str:
        """
        Categorizes ACWR into risk levels
        """
        if pd.isna(acwr):
            return 'Unknown'
        elif acwr < 0.8:
            return 'Very Low'
        elif acwr <= 1.3:
            return 'Optimal'
        elif acwr <= 1.5:
            return 'Elevated'
        else:
            return 'High Risk'


# Algorithm 2: Training Monotony Calculator

class MonotonyCalculator:
    """
    Calculates training monotony
    Motivation:
    - Monotony = Mean weekly load / Standard deviation of weekly loads
    - Monotony values above 2.5 indicate lack of week-to-week variance
    - Lower variation leads to increased injury risk
    - Optimal values are between: 1.5 and 2.0
    """
    def __init__(self, window: int = 7):
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate training monotony.
        
        Algorithm:
        1. Group by athlete
        2. For each rolling 7-day window:
           - Calculate mean load
           - Calculate standard deviation
           - Monotony = mean / std dev
        3. Handle the edge case: if std dev = 0 then set monotony to 0
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        def calculate_monotony_for_group(group):
            rolling_mean = group['session_load'].rolling(window=self.window, min_periods=2).mean()
            rolling_std = group['session_load'].rolling(window=self.window, min_periods=2).std()
            monotony = np.where(rolling_std > 0, rolling_mean / rolling_std, 0)
            return pd.Series(monotony, index=group.index)
        result['training_monotony'] = (result.groupby('athlete_id', group_keys=False).apply(calculate_monotony_for_group))
        return result


# Algorithm 3: Training Strain Calculator

class StrainCalculator:
    """
    Motivation:
    - Strain = Weekly total load * Monotony
    - Accounts for both volume and variation
    - High strain with high monotony leads to increased injury risk
    - Anywhere above 6000 AU is considered a high strain value
    """
    
    def __init__(self, window: int = 7):
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates training strain.
        
        Algorithm:
        1. Calculate cumulative 7-day load
        2. Multiply by training monotony
        3. Strain = Cumulative load × Monotony
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        result['cumulative_load'] = (result.groupby('athlete_id')['session_load'].transform(lambda x: x.rolling(window=self.window, min_periods=1).sum()))
        result['training_strain'] = result['cumulative_load'] * result['training_monotony']
        result['strain_category'] = result['training_strain'].apply(self._categorize_strain)
        return result
    
    @staticmethod
    def _categorize_strain(strain: float) -> str:
        """
        Categorizes strain level.
        """
        if pd.isna(strain):
            return 'Unknown'
        elif strain < 3000:
            return 'Low'
        elif strain < 6000:
            return 'Moderate'
        else:
            return 'High'


# Algorithm 4: Load Spike Detector

class SpikeDetector:
    """
    Detects abnormal load spikes.
    We flag spikes via the following:
    1. Statistical: Load > mean + k*std
    2. Percentage: Load increase > _% from previous week
    3. For ACWR values over 1.5
    """
    
    def __init__(self, std_threshold: float = 2.0, percent_threshold: float = 20):
        self.std_threshold = std_threshold
        self.percent_threshold = percent_threshold
    
    def detect_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects load spikes using multiple criteria
        
        Algorithm:
        1. Calculate z-scores
        2. Detect statistical spikes by comparing the magnitude of the z-score with the threshold
        3. Calculate weekly change
        4. Detect percentage spikes and check it with the threshold value
        5. Flagged if ACWR value is over 1.5
        6. Aggregate flags into one
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        def detect_for_athlete(group):
            group = group.copy()
            mean_load = group['session_load'].mean()
            std_load = group['session_load'].std()
            if std_load > 0:
                group['z_score'] = (group['session_load'] - mean_load) / std_load
                group['statistical_spike'] = group['z_score'].abs() > self.std_threshold
            else:
                group['z_score'] = 0
                group['statistical_spike'] = False
            group['weekly_load'] = group['session_load'].rolling(window=7, min_periods=1).sum()
            group['prev_week_load'] = group['weekly_load'].shift(7)
            group['load_change_percent'] = ((group['weekly_load'] - group['prev_week_load']) / group['prev_week_load'] * 100)
            group['percentage_spike'] = group['load_change_percent'] > self.percent_threshold
            if 'acwr' in group.columns:
                group['acwr_spike'] = group['acwr'] > 1.5
            else:
                group['acwr_spike'] = False
            group['is_spike'] = (group['statistical_spike'] | group['percentage_spike'] | group['acwr_spike'])
            group['spike_severity'] = (group['statistical_spike'].astype(int) + group['percentage_spike'].astype(int) + group['acwr_spike'].astype(int))
            return group
        result = result.groupby('athlete_id', group_keys=False).apply(detect_for_athlete)
        return result


# Algorithm 5: Load Aggregation

class LoadAggregator:
    """
    Flexible load aggregation across time windows.
    
    Supports custom time periods with multiple metrics (sum, mean, max, std, count)
    """
    
    @staticmethod
    def aggregate(
        df: pd.DataFrame,
        period: str = 'weekly',
        metrics: List[str] = ['sum', 'mean', 'max'],
        group_by_athlete: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate loads over specified periods.
        
        Algorithm:
        1. Convert period to pandas frequency
        2. Set date as index
        3. Group by athlete (if requested)
        4. Resample to target period
        5. Apply aggregation functions
        
        Specified input:
            group_by_athlete: Whether to group by athlete
        """
        period_map = {'daily': 'D', 'weekly': 'W', 'biweekly': '2W', 'monthly': 'M', 'quarterly': 'Q'}
        freq = period_map.get(period, period)
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        if group_by_athlete:
            aggregated = (result.set_index('date').groupby('athlete_id')['session_load'].resample(freq).agg(metrics).reset_index())
        else:
            aggregated = (result.set_index('date')['session_load'].resample(freq).agg(metrics).reset_index())
        if freq == 'W':
            aggregated['week_number'] = aggregated['date'].dt.isocalendar().week
            aggregated['year'] = aggregated['date'].dt.year
        elif freq == 'M':
            aggregated['month'] = aggregated['date'].dt.month
            aggregated['year'] = aggregated['date'].dt.year
        return aggregated


# How to use algorithms (with example)

if __name__ == "__main__":
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    base_load = 500
    loads = []
    for i in range(n_days):
        day_of_week = i % 7
        if day_of_week == 6:
            load = 0
        elif day_of_week in [1, 3]:
            load = base_load * 0.4 + np.random.normal(0, 50)
        elif day_of_week in [0, 2, 4]:
            load = base_load * 1.2 + np.random.normal(0, 80)
        else:
            load = base_load * 0.8 + np.random.normal(0, 60)
        loads.append(max(0, load))
    
    sample_df = pd.DataFrame({'athlete_id': ['ATH_001'] * n_days, 'date': dates, 'session_load': loads})
    acwr_calc = ACWRCalculator()
    df_acwr = acwr_calc.calculate(sample_df, method='rolling_average')
    print(f"Mean ACWR: {df_acwr['acwr'].mean()}")
    print(f"ACWR range: {df_acwr['acwr'].min()} - {df_acwr['acwr'].max()}")
    print(f"Sample ACWR values:")
    print(df_acwr[['date', 'session_load', 'acute_load', 'chronic_load', 'acwr', 'acwr_category']].tail(10).to_string(index=False))
    monotony_calc = MonotonyCalculator()
    df_monotony = monotony_calc.calculate(df_acwr)
    print(f"Mean monotony: {df_monotony['training_monotony'].mean()}")
    print(f"Max monotony: {df_monotony['training_monotony'].max()}")
    strain_calc = StrainCalculator()
    df_strain = strain_calc.calculate(df_monotony)
    print(f"Mean strain: {df_strain['training_strain'].mean()} AU")
    print(f"Max strain: {df_strain['training_strain'].max()} AU")
    spike_detector = SpikeDetector(std_threshold=2.0, percent_threshold=20)
    df_spikes = spike_detector.detect_spikes(df_strain)
    num_spikes = df_spikes['is_spike'].sum()
    print(f"Detected {num_spikes} spikes")
    if num_spikes > 0:
        print(f"Spike details:")
        spike_data = df_spikes[df_spikes['is_spike']][['date', 'session_load', 'acwr', 'load_change_percent', 'spike_severity']]
        print(spike_data.head(5).to_string(index=False))
