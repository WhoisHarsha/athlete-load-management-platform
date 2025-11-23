"""
=============================================================================
PERFORMANCE TREND ANALYSIS ALGORITHMS
=============================================================================
Purpose: Statistical trend analysis for training loads and performance
Author: Samuel Greeman (Sports Data Science)
Date: November 2025

Algorithms:
- Linear regression trend detection
- Change point detection
- Performance forecasting
- Fatigue accumulation tracking

Time Complexity: O(n) for trend analysis
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ALGORITHM 5: TREND ANALYZER
# =============================================================================
# Time Complexity: O(n) for linear regression
# =============================================================================

class TrendAnalyzer:
    """
    Statistical trend analysis for training loads.
    
    Methods:
    - Linear regression for trend detection
    - Mann-Kendall test for monotonic trends
    - Change point detection
    - Performance forecasting
    """
    
    @staticmethod
    def analyze_trends(df: pd.DataFrame, athlete_id: str) -> Dict:
        """
        Comprehensive trend analysis for an athlete.
        
        Algorithm:
        1. Linear Regression: y = mx + b
           - Slope (m) indicates trend direction
           - R² indicates trend strength
           - p-value indicates significance
        2. Calculate volatility (standard deviation)
        3. Compare recent vs baseline performance
        4. Detect acceleration/deceleration in trend
        
        Args:
            df: DataFrame with loads
            athlete_id: Athlete to analyze
            
        Returns:
            Dictionary with trend metrics
            
        Time Complexity: O(n) where n = sessions for this athlete
        """
        athlete_data = df[df['athlete_id'] == athlete_id].copy()
        athlete_data = athlete_data.sort_values('date')
        
        loads = athlete_data['session_load'].values
        n = len(loads)
        
        if n < 7:
            return {'error': 'Insufficient data (minimum 7 sessions required)'}
        
        # Linear regression
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, loads)
        
        # Determine trend direction
        if p_value < 0.05:
            direction = 'increasing' if slope > 0 else 'decreasing'
            significance = 'significant'
        else:
            direction = 'stable'
            significance = 'not significant'
        
        # Volatility metrics
        volatility = np.std(loads)
        mean_load = np.mean(loads)
        cv = (volatility / mean_load * 100) if mean_load > 0 else 0
        
        # Baseline vs recent comparison
        baseline_window = min(7, n // 3)
        baseline_load = np.mean(loads[:baseline_window])
        recent_load = np.mean(loads[-7:])
        percent_change = ((recent_load - baseline_load) / baseline_load * 100) if baseline_load > 0 else 0
        
        # Trend acceleration (is trend speeding up or slowing down?)
        if n >= 14:
            first_half_slope = stats.linregress(x[:n//2], loads[:n//2])[0]
            second_half_slope = stats.linregress(x[n//2:], loads[n//2:])[0]
            acceleration = second_half_slope - first_half_slope
            
            if abs(acceleration) > 1:
                accel_status = 'accelerating' if acceleration > 0 else 'decelerating'
            else:
                accel_status = 'constant'
        else:
            acceleration = 0
            accel_status = 'insufficient data'
        
        return {
            'athlete_id': athlete_id,
            'data_points': n,
            'mean_load': round(mean_load, 2),
            'trend_direction': direction,
            'trend_significance': significance,
            'trend_slope': round(slope, 4),
            'slope_per_week': round(slope * 7, 2),
            'p_value': round(p_value, 4),
            'r_squared': round(r_value**2, 4),
            'volatility': round(volatility, 2),
            'coefficient_variation_percent': round(cv, 2),
            'baseline_load': round(baseline_load, 2),
            'recent_load': round(recent_load, 2),
            'percent_change': round(percent_change, 2),
            'trend_acceleration': round(acceleration, 4),
            'acceleration_status': accel_status
        }
    
    @staticmethod
    def detect_change_points(loads: np.ndarray, threshold: float = 2.0) -> List[int]:
        """
        Detect significant change points in load series.
        
        Algorithm (Simplified PELT):
        1. Calculate 7-day moving average
        2. Compute derivative (rate of change)
        3. Find points where |change| > threshold * std
        4. Return indices of change points
        
        Args:
            loads: Array of session loads
            threshold: Standard deviations for change detection
            
        Returns:
            List of indices where significant changes occur
            
        Time Complexity: O(n)
        """
        if len(loads) < 10:
            return []
        
        # Calculate moving average
        window = 7
        ma = np.convolve(loads, np.ones(window)/window, mode='valid')
        
        # Calculate derivative (change)
        diff = np.diff(ma)
        std = np.std(loads)
        
        # Find change points
        change_points = np.where(np.abs(diff) > threshold * std)[0]
        
        return change_points.tolist()
    
    @staticmethod
    def forecast_load(df: pd.DataFrame, athlete_id: str, days_ahead: int = 7) -> Dict:
        """
        Simple load forecast using linear regression.
        
        Algorithm:
        1. Fit linear model to historical data
        2. Extrapolate for future days
        3. Calculate prediction interval
        
        Args:
            df: DataFrame with historical loads
            athlete_id: Athlete to forecast
            days_ahead: Number of days to forecast
            
        Returns:
            Dictionary with forecast data
        """
        athlete_data = df[df['athlete_id'] == athlete_id].copy()
        athlete_data = athlete_data.sort_values('date')
        
        loads = athlete_data['session_load'].values
        n = len(loads)
        
        if n < 14:
            return {'error': 'Insufficient data for forecasting'}
        
        # Fit linear model
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, loads)
        
        # Generate forecast
        future_x = np.arange(n, n + days_ahead)
        forecast = slope * future_x + intercept
        
        # Prediction interval (95% confidence)
        residuals = loads - (slope * x + intercept)
        residual_std = np.std(residuals)
        margin = 1.96 * residual_std
        
        forecast_upper = forecast + margin
        forecast_lower = np.maximum(0, forecast - margin)
        
        return {
            'athlete_id': athlete_id,
            'forecast_days': days_ahead,
            'predicted_loads': forecast.tolist(),
            'upper_bound': forecast_upper.tolist(),
            'lower_bound': forecast_lower.tolist(),
            'trend_slope': round(slope, 2),
            'confidence': round((1 - p_value) * 100, 2)
        }


class FatigueTracker:
    """
    Track cumulative fatigue over time.
    
    Concept: Fitness-Fatigue Model (Banister)
    - Fitness builds slowly (chronic adaptation)
    - Fatigue accumulates quickly (acute stress)
    - Performance = Fitness - Fatigue
    """
    
    @staticmethod
    def calculate_fatigue_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative fatigue index.
        
        Algorithm:
        1. Exponential decay model
        2. Fatigue(t) = Σ Load(i) * exp(-(t-i)/τ)
        3. τ = time constant (7 days for fatigue)
        
        Args:
            df: DataFrame with session loads
            
        Returns:
            DataFrame with fatigue_index column
        """
        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values(['athlete_id', 'date'])
        
        def calculate_for_athlete(group):
            group = group.copy()
            n = len(group)
            fatigue = np.zeros(n)
            
            tau_fatigue = 7
            
            for i in range(n):
                for j in range(i + 1):
                    days_ago = i - j
                    decay = np.exp(-days_ago / tau_fatigue)
                    fatigue[i] += group.iloc[j]['session_load'] * decay
            
            group['fatigue_index'] = fatigue
            return group
        
        result = result.groupby('athlete_id', group_keys=False).apply(calculate_for_athlete)
        
        return result


if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE TREND ANALYSIS - DEMONSTRATION")
    print("=" * 80)
    
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    loads = []
    for i in range(n_days):
        trend = i * 2
        noise = np.random.normal(0, 50)
        load = max(0, 400 + trend + noise)
        loads.append(load)
    
    sample_df = pd.DataFrame({
        'athlete_id': ['ATH_001'] * n_days,
        'date': dates,
        'session_load': loads
    })
    
    print("\n[1/3] Analyzing Performance Trends...")
    trends = TrendAnalyzer.analyze_trends(sample_df, 'ATH_001')
    
    print(f"\n   Trend Analysis Results:")
    for key, value in trends.items():
        print(f"   • {key}: {value}")
    
    print("\n[2/3] Detecting Change Points...")
    change_points = TrendAnalyzer.detect_change_points(loads, threshold=2.0)
    print(f"   ✓ Detected {len(change_points)} change points")
    
    print("\n[3/3] Forecasting Future Load...")
    forecast = TrendAnalyzer.forecast_load(sample_df, 'ATH_001', days_ahead=7)
    
    if 'error' not in forecast:
        print(f"\n   7-Day Forecast: {forecast['predicted_loads']}")
    
    print("\n✅ TREND ANALYSIS COMPLETE")