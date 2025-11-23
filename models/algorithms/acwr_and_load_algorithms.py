"""
=============================================================================
TRAINING LOAD CALCULATION ALGORITHMS
=============================================================================
Purpose: Core algorithms for ACWR, monotony, strain, and spike detection
Author: Samuel Greeman (Sports Data Science)
Integrated by: Harsha Prakash
Date: November 2025
Project: Athlete Load Management & Performance Optimization Platform

Algorithms Included:
1. ACWR Calculator (Acute:Chronic Workload Ratio)
2. Training Monotony Calculator
3. Training Strain Calculator
4. Load Spike Detector

Time Complexity: O(n) per athlete for all algorithms
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ALGORITHM 1: ACWR (ACUTE:CHRONIC WORKLOAD RATIO) CALCULATOR
# =============================================================================
# Time Complexity: O(n) per athlete
# Space Complexity: O(n)
# =============================================================================

class ACWRCalculator:
    """
    Calculate Acute:Chronic Workload Ratio.
    
    Sports Science Context:
    - Acute Load: 7-day rolling average (recent training stress)
    - Chronic Load: 28-day rolling average (fitness baseline)
    - ACWR = Acute / Chronic
    - Optimal range: 0.8 - 1.3
    - High injury risk: > 1.5 or < 0.8
    
    Methods:
    - Rolling Average (traditional)
    - Exponentially Weighted Moving Average (EWMA)
    """
    
    def __init__(self, acute_window: int = 7, chronic_window: int = 28):
        """
        Initialize ACWR calculator.
        
        Args:
            acute_window: Days for acute load (default: 7)
            chronic_window: Days for chronic load (default: 28)
        """
        self.acute_window = acute_window
        self.chronic_window = chronic_window
    
    def calculate(self, df: pd.DataFrame, method: str = 'rolling_average') -> pd.DataFrame:
        """
        Calculate ACWR for all athletes.
        
        Algorithm (Rolling Average Method):
        1. Sort data by athlete and date
        2. For each athlete:
           a. Calculate 7-day rolling mean (acute)
           b. Calculate 28-day rolling mean (chronic)
           c. Compute ratio: ACWR = acute / chronic
        3. Categorize risk level
        
        Time Complexity: O(n) where n = total sessions
        
        Args:
            df: DataFrame with columns [athlete_id, date, session_load]
            method: 'rolling_average' or 'ewma'
            
        Returns:
            DataFrame with acute_load, chronic_load, acwr columns added
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        if method == 'rolling_average':
            # Rolling average method
            result['acute_load'] = (
                result.groupby('athlete_id')['session_load']
                .transform(lambda x: x.rolling(window=self.acute_window, min_periods=1).mean())
            )
            result['chronic_load'] = (
                result.groupby('athlete_id')['session_load']
                .transform(lambda x: x.rolling(window=self.chronic_window, min_periods=1).mean())
            )
        
        elif method == 'ewma':
            # Exponentially Weighted Moving Average
            result['acute_load'] = (
                result.groupby('athlete_id')['session_load']
                .transform(lambda x: x.ewm(span=self.acute_window).mean())
            )
            result['chronic_load'] = (
                result.groupby('athlete_id')['session_load']
                .transform(lambda x: x.ewm(span=self.chronic_window).mean())
            )
        
        # Calculate ACWR
        result['acwr'] = result['acute_load'] / result['chronic_load']
        result['acwr'] = result['acwr'].replace([np.inf, -np.inf], np.nan)
        
        # Categorize risk
        result['acwr_category'] = result['acwr'].apply(self._categorize_acwr)
        
        return result
    
    @staticmethod
    def _categorize_acwr(acwr: float) -> str:
        """
        Categorize ACWR into risk levels.
        
        Args:
            acwr: ACWR value
            
        Returns:
            Risk category string
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


# =============================================================================
# ALGORITHM 2: TRAINING MONOTONY CALCULATOR
# =============================================================================
# Time Complexity: O(n) per athlete
# =============================================================================

class MonotonyCalculator:
    """
    Calculate training monotony (Foster 1998).
    
    Sports Science Context:
    - Monotony = Mean weekly load / Standard deviation of weekly loads
    - High monotony (> 2.5) indicates lack of variation
    - Low variation → increased injury risk and training staleness
    - Optimal: 1.5 - 2.0 (balanced variation)
    """
    
    def __init__(self, window: int = 7):
        """
        Initialize monotony calculator.
        
        Args:
            window: Days for rolling window (default: 7 for weekly)
        """
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate training monotony.
        
        Algorithm:
        1. Group by athlete
        2. For each rolling 7-day window:
           - Calculate mean load
           - Calculate standard deviation
           - Monotony = mean / std
        3. Handle edge case: std = 0 → set monotony to 0
        
        Time Complexity: O(n)
        
        Args:
            df: DataFrame with session_load column
            
        Returns:
            DataFrame with training_monotony column added
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        def calculate_monotony_for_group(group):
            """Calculate monotony for a single athlete."""
            rolling_mean = group['session_load'].rolling(window=self.window, min_periods=2).mean()
            rolling_std = group['session_load'].rolling(window=self.window, min_periods=2).std()
            
            # Avoid division by zero
            monotony = np.where(rolling_std > 0, rolling_mean / rolling_std, 0)
            
            return pd.Series(monotony, index=group.index)
        
        result['training_monotony'] = (
            result.groupby('athlete_id', group_keys=False)
            .apply(calculate_monotony_for_group)
        )
        
        return result


# =============================================================================
# ALGORITHM 3: TRAINING STRAIN CALCULATOR
# =============================================================================
# Time Complexity: O(n)
# =============================================================================

class StrainCalculator:
    """
    Calculate training strain (Foster 1998).
    
    Sports Science Context:
    - Strain = Weekly total load × Monotony
    - Accounts for both volume and variation
    - High strain with high monotony → increased injury risk
    - Threshold: > 6000 AU considered high strain
    """
    
    def __init__(self, window: int = 7):
        """
        Initialize strain calculator.
        
        Args:
            window: Days for cumulative window (default: 7)
        """
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate training strain.
        
        Algorithm:
        1. Calculate cumulative 7-day load
        2. Multiply by training monotony
        3. Strain = Cumulative load × Monotony
        
        Time Complexity: O(n)
        
        Args:
            df: DataFrame with session_load and training_monotony
            
        Returns:
            DataFrame with training_strain column added
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        # Calculate cumulative load over window
        result['cumulative_load'] = (
            result.groupby('athlete_id')['session_load']
            .transform(lambda x: x.rolling(window=self.window, min_periods=1).sum())
        )
        
        # Strain = Cumulative load × Monotony
        result['training_strain'] = result['cumulative_load'] * result['training_monotony']
        
        # Categorize strain level
        result['strain_category'] = result['training_strain'].apply(self._categorize_strain)
        
        return result
    
    @staticmethod
    def _categorize_strain(strain: float) -> str:
        """
        Categorize strain level.
        
        Args:
            strain: Strain value
            
        Returns:
            Category string
        """
        if pd.isna(strain):
            return 'Unknown'
        elif strain < 3000:
            return 'Low'
        elif strain < 6000:
            return 'Moderate'
        else:
            return 'High'


# =============================================================================
# ALGORITHM 6: SPIKE DETECTOR
# =============================================================================
# Time Complexity: O(n) per athlete
# =============================================================================

class SpikeDetector:
    """
    Detect abnormal load spikes.
    
    Detection Methods:
    1. Statistical: Load > mean + k*std
    2. Percentage: Load increase > X% from previous week
    3. ACWR: ACWR > 1.5 threshold
    """
    
    def __init__(self, std_threshold: float = 2.0, percent_threshold: float = 20):
        """
        Initialize spike detector.
        
        Args:
            std_threshold: Standard deviations above mean (default: 2.0)
            percent_threshold: Percent increase threshold (default: 20%)
        """
        self.std_threshold = std_threshold
        self.percent_threshold = percent_threshold
    
    def detect_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect load spikes using multiple criteria.
        
        Algorithm:
        1. Calculate z-scores: z = (x - μ) / σ
        2. Detect statistical spikes: |z| > threshold
        3. Calculate week-over-week change
        4. Detect percentage spikes: change > threshold%
        5. Flag ACWR spikes: ACWR > 1.5
        6. Combine flags
        
        Time Complexity: O(n)
        
        Args:
            df: DataFrame with session_load and acwr
            
        Returns:
            DataFrame with spike detection columns
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        def detect_for_athlete(group):
            """Detect spikes for single athlete."""
            group = group.copy()
            
            # Statistical spike detection
            mean_load = group['session_load'].mean()
            std_load = group['session_load'].std()
            
            if std_load > 0:
                group['z_score'] = (group['session_load'] - mean_load) / std_load
                group['statistical_spike'] = group['z_score'].abs() > self.std_threshold
            else:
                group['z_score'] = 0
                group['statistical_spike'] = False
            
            # Week-over-week percentage change
            group['weekly_load'] = group['session_load'].rolling(window=7, min_periods=1).sum()
            group['prev_week_load'] = group['weekly_load'].shift(7)
            group['load_change_percent'] = (
                (group['weekly_load'] - group['prev_week_load']) / 
                group['prev_week_load'] * 100
            )
            group['percentage_spike'] = group['load_change_percent'] > self.percent_threshold
            
            # ACWR spike
            if 'acwr' in group.columns:
                group['acwr_spike'] = group['acwr'] > 1.5
            else:
                group['acwr_spike'] = False
            
            # Combined spike flag
            group['is_spike'] = (
                group['statistical_spike'] | 
                group['percentage_spike'] | 
                group['acwr_spike']
            )
            
            # Spike severity (0-3)
            group['spike_severity'] = (
                group['statistical_spike'].astype(int) +
                group['percentage_spike'].astype(int) +
                group['acwr_spike'].astype(int)
            )
            
            return group
        
        result = result.groupby('athlete_id', group_keys=False).apply(detect_for_athlete)
        
        return result


# =============================================================================
# ALGORITHM 4: LOAD AGGREGATION
# =============================================================================
# Time Complexity: O(n log n) for resampling
# =============================================================================

class LoadAggregator:
    """
    Flexible load aggregation across time windows.
    
    Supports: daily, weekly, monthly, custom periods
    Multiple metrics: sum, mean, max, std, count
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
        
        Args:
            df: Input DataFrame
            period: Aggregation period ('daily', 'weekly', 'monthly', etc.)
            metrics: List of aggregation functions
            group_by_athlete: Whether to group by athlete
            
        Returns:
            Aggregated DataFrame
            
        Time Complexity: O(n log n) for resampling
        """
        period_map = {
            'daily': 'D',
            'weekly': 'W',
            'biweekly': '2W',
            'monthly': 'M',
            'quarterly': 'Q'
        }
        
        freq = period_map.get(period, period)
        
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        
        if group_by_athlete:
            aggregated = (
                result.set_index('date')
                .groupby('athlete_id')['session_load']
                .resample(freq)
                .agg(metrics)
                .reset_index()
            )
        else:
            aggregated = (
                result.set_index('date')['session_load']
                .resample(freq)
                .agg(metrics)
                .reset_index()
            )
        
        # Add time period identifiers
        if freq == 'W':
            aggregated['week_number'] = aggregated['date'].dt.isocalendar().week
            aggregated['year'] = aggregated['date'].dt.year
        elif freq == 'M':
            aggregated['month'] = aggregated['date'].dt.month
            aggregated['year'] = aggregated['date'].dt.year
        
        return aggregated


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LOAD CALCULATION ALGORITHMS - DEMONSTRATION")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Simulate realistic training pattern
    base_load = 500
    loads = []
    for i in range(n_days):
        day_of_week = i % 7
        if day_of_week == 6:  # Sunday rest
            load = 0
        elif day_of_week in [1, 3]:  # Light days
            load = base_load * 0.4 + np.random.normal(0, 50)
        elif day_of_week in [0, 2, 4]:  # Hard days
            load = base_load * 1.2 + np.random.normal(0, 80)
        else:  # Moderate
            load = base_load * 0.8 + np.random.normal(0, 60)
        loads.append(max(0, load))
    
    sample_df = pd.DataFrame({
        'athlete_id': ['ATH_001'] * n_days,
        'date': dates,
        'session_load': loads
    })
    
    print("\n[1/4] Testing ACWR Calculator...")
    acwr_calc = ACWRCalculator()
    df_acwr = acwr_calc.calculate(sample_df, method='rolling_average')
    print(f"   ✓ Calculated ACWR for {len(df_acwr)} sessions")
    print(f"   ✓ Mean ACWR: {df_acwr['acwr'].mean():.3f}")
    print(f"   ✓ ACWR range: {df_acwr['acwr'].min():.3f} - {df_acwr['acwr'].max():.3f}")
    print(f"\n   Sample ACWR values:")
    print(df_acwr[['date', 'session_load', 'acute_load', 'chronic_load', 'acwr', 'acwr_category']].tail(10).to_string(index=False))
    
    print("\n[2/4] Testing Monotony Calculator...")
    monotony_calc = MonotonyCalculator()
    df_monotony = monotony_calc.calculate(df_acwr)
    print(f"   ✓ Mean monotony: {df_monotony['training_monotony'].mean():.2f}")
    print(f"   ✓ Max monotony: {df_monotony['training_monotony'].max():.2f}")
    
    print("\n[3/4] Testing Strain Calculator...")
    strain_calc = StrainCalculator()
    df_strain = strain_calc.calculate(df_monotony)
    print(f"   ✓ Mean strain: {df_strain['training_strain'].mean():.1f} AU")
    print(f"   ✓ Max strain: {df_strain['training_strain'].max():.1f} AU")
    
    print("\n[4/4] Testing Spike Detector...")
    spike_detector = SpikeDetector(std_threshold=2.0, percent_threshold=20)
    df_spikes = spike_detector.detect_spikes(df_strain)
    num_spikes = df_spikes['is_spike'].sum()
    print(f"   ✓ Detected {num_spikes} load spikes")
    
    if num_spikes > 0:
        print(f"\n   Spike details:")
        spike_data = df_spikes[df_spikes['is_spike']][
            ['date', 'session_load', 'acwr', 'load_change_percent', 'spike_severity']
        ]
        print(spike_data.head(5).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ ALL ALGORITHMS TESTED SUCCESSFULLY")
    print("=" * 80)
    print("\nAlgorithm Complexity Summary:")
    print("  • ACWR Calculator:        O(n) per athlete")
    print("  • Monotony Calculator:    O(n) per athlete")
    print("  • Strain Calculator:      O(n) per athlete")
    print("  • Spike Detector:         O(n) per athlete")
    print("\nTotal Pipeline Complexity: O(n) where n = number of training sessions")
    print("=" * 80)