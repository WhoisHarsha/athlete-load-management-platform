"""
=============================================================================
COMPLETE ALGORITHM DEMONSTRATION AND TESTING
=============================================================================
Purpose: Demonstrate all training load algorithms with real data
Author: Samuel Greeman (Sports Data Science)
Date: November 2025

This script:
1. Loads training session data
2. Runs all load calculation algorithms
3. Generates comprehensive visualizations
4. Outputs performance metrics
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List
import os
import warnings
warnings.filterwarnings('ignore')

# Import algorithm classes
from acwr_and_load_algorithms import (
    ACWRCalculator,
    MonotonyCalculator,
    StrainCalculator,
    SpikeDetector,
    LoadAggregator
)
from performance_trends import TrendAnalyzer, FatigueTracker
from load_visualizations import LoadVisualizer

print("=" * 80)
print("ATHLETE LOAD MANAGEMENT - COMPLETE ALGORITHM PIPELINE")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1/7] Loading training session data...")

try:
    # Load from database exports
    df_sessions = pd.read_csv('../integration/training_sessions.csv')
    df_loads = pd.read_csv('../integration/load_calculations.csv')
    
    # Merge data
    df = df_sessions[['session_id', 'athlete_id', 'session_date', 'session_type', 
                      'duration_minutes', 'intensity_level', 'rpe']].copy()
    df = df.rename(columns={'session_date': 'date'})
    
    # Add load calculations
    df = df.merge(df_loads[['session_id', 'session_load', 'acute_load', 
                            'chronic_load', 'acwr', 'training_monotony', 
                            'training_strain']], 
                  on='session_id', how='left')
    
    print(f"   ✓ Loaded {len(df)} training sessions")
    print(f"   ✓ Athletes: {df['athlete_id'].nunique()}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    using_real_data = True

except FileNotFoundError:
    print("   ⚠ Database exports not found, generating sample data...")
    
    # Generate sample data
    np.random.seed(42)
    n_athletes = 5
    n_days_per_athlete = 60
    
    all_data = []
    session_id_counter = 1
    
    for ath in range(1, n_athletes + 1):
        athlete_id = f"ATH_{str(ath).zfill(3)}"
        dates = pd.date_range('2024-01-01', periods=n_days_per_athlete, freq='D')
        
        for i, date in enumerate(dates):
            day_of_week = i % 7
            
            if day_of_week == 6:  # Sunday rest
                continue
            
            # Vary load by day
            if day_of_week in [0, 2, 4]:  # Hard days
                load = np.random.normal(600, 100)
                intensity = np.random.uniform(7, 9)
                session_type = np.random.choice(['HIIT', 'Match/Competition', 'Strength'])
            elif day_of_week in [1, 3]:  # Light days
                load = np.random.normal(300, 80)
                intensity = np.random.uniform(4, 6)
                session_type = np.random.choice(['Skills', 'Recovery'])
            else:  # Moderate
                load = np.random.normal(450, 90)
                intensity = np.random.uniform(5, 7)
                session_type = np.random.choice(['Endurance', 'Speed'])
            
            duration = load / (intensity * 10)
            
            all_data.append({
                'session_id': f"SES_{str(session_id_counter).zfill(5)}",
                'athlete_id': athlete_id,
                'date': date,
                'session_type': session_type,
                'duration_minutes': max(30, duration),
                'intensity_level': intensity,
                'rpe': intensity,
                'session_load': max(0, load)
            })
            session_id_counter += 1
    
    df = pd.DataFrame(all_data)
    print(f"   ✓ Generated {len(df)} sample sessions")
    print(f"   ✓ Athletes: {df['athlete_id'].nunique()}")
    
    using_real_data = False

# =============================================================================
# STEP 2: CALCULATE ACWR
# =============================================================================
print("\n[2/7] Calculating ACWR (Acute:Chronic Workload Ratio)...")

if not using_real_data or df['acwr'].isna().all():
    acwr_calc = ACWRCalculator(acute_window=7, chronic_window=28)
    df = acwr_calc.calculate(df, method='rolling_average')
    print(f"   ✓ ACWR calculated using rolling average method")
else:
    print(f"   ✓ Using existing ACWR values from database")

print(f"   ✓ Mean ACWR: {df['acwr'].mean():.3f}")
print(f"   ✓ High risk sessions (ACWR > 1.5): {(df['acwr'] > 1.5).sum()}")

# =============================================================================
# STEP 3: CALCULATE TRAINING MONOTONY
# =============================================================================
print("\n[3/7] Calculating Training Monotony...")

if not using_real_data or df['training_monotony'].isna().all():
    monotony_calc = MonotonyCalculator(window=7)
    df = monotony_calc.calculate(df)
    print(f"   ✓ Monotony calculated using 7-day window")
else:
    print(f"   ✓ Using existing monotony values from database")

print(f"   ✓ Mean monotony: {df['training_monotony'].mean():.2f}")
print(f"   ✓ High monotony sessions (> 2.5): {(df['training_monotony'] > 2.5).sum()}")

# =============================================================================
# STEP 4: CALCULATE TRAINING STRAIN
# =============================================================================
print("\n[4/7] Calculating Training Strain...")

if not using_real_data or df['training_strain'].isna().all():
    strain_calc = StrainCalculator(window=7)
    df = strain_calc.calculate(df)
    print(f"   ✓ Strain calculated")
else:
    print(f"   ✓ Using existing strain values from database")

print(f"   ✓ Mean strain: {df['training_strain'].mean():.1f} AU")

# =============================================================================
# STEP 5: DETECT LOAD SPIKES
# =============================================================================
print("\n[5/7] Detecting Load Spikes...")

spike_detector = SpikeDetector(std_threshold=2.0, percent_threshold=20)
df = spike_detector.detect_spikes(df)

num_spikes = df['is_spike'].sum()
print(f"   ✓ Detected {num_spikes} load spikes")
print(f"   ✓ Spike rate: {num_spikes / len(df) * 100:.1f}%")

# =============================================================================
# STEP 6: ANALYZE TRENDS
# =============================================================================
print("\n[6/7] Analyzing Performance Trends...")

# Analyze trends for top 3 athletes by total load
top_athletes = df.groupby('athlete_id')['session_load'].sum().nlargest(3).index

trend_results = []
for athlete_id in top_athletes:
    trends = TrendAnalyzer.analyze_trends(df, athlete_id)
    if 'error' not in trends:
        trend_results.append(trends)

if trend_results:
    trends_df = pd.DataFrame(trend_results)
    print(f"   ✓ Analyzed trends for {len(trends_df)} athletes")
    print(f"\n   Trend Summary:")
    print(trends_df[['athlete_id', 'trend_direction', 'slope_per_week', 
                     'r_squared', 'percent_change']].to_string(index=False))

# =============================================================================
# STEP 7: GENERATE VISUALIZATIONS
# =============================================================================
print("\n[7/7] Generating Visualizations...")

# Create output directory
os.makedirs('algorithm_outputs', exist_ok=True)

# Individual athlete dashboard
sample_athlete = df['athlete_id'].iloc[0]
LoadVisualizer.create_dashboard(df, sample_athlete, 
                                'algorithm_outputs/athlete_dashboard.png')

# Team comparison
LoadVisualizer.create_team_comparison(df, 
                                     'algorithm_outputs/team_comparison.png')

# =============================================================================
# STEP 8: GENERATE SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 80)
print("ALGORITHM PIPELINE SUMMARY")
print("=" * 80)

print(f"\nData Overview:")
print(f"  • Total Sessions: {len(df)}")
print(f"  • Athletes: {df['athlete_id'].nunique()}")
print(f"  • Date Range: {df['date'].min()} to {df['date'].max()}")

print(f"\nACWR Statistics:")
print(f"  • Mean: {df['acwr'].mean():.3f}")
print(f"  • Optimal Range (0.8-1.3): {((df['acwr'] >= 0.8) & (df['acwr'] <= 1.3)).sum()} sessions ({((df['acwr'] >= 0.8) & (df['acwr'] <= 1.3)).sum() / len(df) * 100:.1f}%)")
print(f"  • High Risk (>1.5): {(df['acwr'] > 1.5).sum()} sessions ({(df['acwr'] > 1.5).sum() / len(df) * 100:.1f}%)")

print(f"\nTraining Monotony:")
print(f"  • Mean: {df['training_monotony'].mean():.2f}")
print(f"  • High Monotony (>2.5): {(df['training_monotony'] > 2.5).sum()} sessions")

print(f"\nTraining Strain:")
print(f"  • Mean: {df['training_strain'].mean():.1f} AU")
print(f"  • High Strain (>6000): {(df['training_strain'] > 6000).sum()} sessions")

print(f"\nLoad Spikes:")
print(f"  • Total Spikes Detected: {df['is_spike'].sum()}")
print(f"  • Severity 3 (all criteria): {(df['spike_severity'] == 3).sum()}")
print(f"  • Severity 2: {(df['spike_severity'] == 2).sum()}")
print(f"  • Severity 1: {(df['spike_severity'] == 1).sum()}")

print("\n" + "=" * 80)
print("✅ ALGORITHM PIPELINE COMPLETE!")
print("=" * 80)

print("\nOutput Files:")
print("  • algorithm_outputs/athlete_dashboard.png")
print("  • algorithm_outputs/team_comparison.png")

print("\nAlgorithm Complexity Analysis:")
print("  • ACWR Calculator:        O(n) per athlete")
print("  • Monotony Calculator:    O(n) per athlete")
print("  • Strain Calculator:      O(n) per athlete")
print("  • Spike Detector:         O(n) per athlete")
print("  • Trend Analyzer:         O(n) per athlete")
print("  • Visualization:          O(n log n) for sorting")
print("\n  Total Pipeline: O(n log n) where n = total training sessions")

print("=" * 80)