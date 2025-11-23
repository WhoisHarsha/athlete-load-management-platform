"""
=============================================================================
TRAINING LOAD VISUALIZATION GENERATOR
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class LoadVisualizer:
    """Generate comprehensive load management visualizations."""
    
    @staticmethod
    def create_dashboard(df: pd.DataFrame, athlete_id: str, output_file: str = 'load_dashboard.png'):
        """Create 6-panel dashboard for an athlete."""
        data = df[df['athlete_id'] == athlete_id].copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        if len(data) == 0:
            print(f"No data for {athlete_id}")
            return None
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Load Management Dashboard - {athlete_id}', fontsize=16, fontweight='bold')
        
        # Plot 1: Session Load
        axes[0, 0].plot(data['date'], data['session_load'], marker='o', linewidth=1, markersize=4, alpha=0.7, color='#3498db')
        axes[0, 0].fill_between(data['date'], data['session_load'], alpha=0.3, color='#3498db')
        axes[0, 0].set_title('Session Load Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Load (AU)')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: ACWR
        if 'acwr' in data.columns:
            axes[0, 1].plot(data['date'], data['acwr'], marker='s', linewidth=2, markersize=5, color='#9b59b6')
            axes[0, 1].axhline(y=1.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='High Risk (1.5)')
            axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Low Risk (0.8)')
            axes[0, 1].fill_between(data['date'], 0.8, 1.3, alpha=0.2, color='green', label='Optimal Zone')
            axes[0, 1].set_title('ACWR Tracking', fontweight='bold')
            axes[0, 1].set_ylabel('ACWR')
            axes[0, 1].legend(fontsize=8)
            axes[0, 1].grid(alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylim(0, 2.5)
        
        # Plot 3: Acute vs Chronic
        if 'acute_load' in data.columns and 'chronic_load' in data.columns:
            axes[1, 0].plot(data['date'], data['acute_load'], label='Acute (7d)', linewidth=2, color='#e74c3c')
            axes[1, 0].plot(data['date'], data['chronic_load'], label='Chronic (28d)', linewidth=2, color='#27ae60')
            axes[1, 0].fill_between(data['date'], data['acute_load'], data['chronic_load'], alpha=0.3, color='gray')
            axes[1, 0].set_title('Acute vs Chronic Load', fontweight='bold')
            axes[1, 0].set_ylabel('Load (AU)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Training Monotony
        if 'training_monotony' in data.columns:
            axes[1, 1].plot(data['date'], data['training_monotony'], marker='d', linewidth=2, markersize=4, color='#f39c12')
            axes[1, 1].axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='High Risk (>2.5)')
            axes[1, 1].axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Moderate (2.0)')
            axes[1, 1].set_title('Training Monotony', fontweight='bold')
            axes[1, 1].set_ylabel('Monotony Index')
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 5: Load Distribution
        axes[2, 0].hist(data['session_load'], bins=20, alpha=0.7, edgecolor='black', color='#3498db')
        axes[2, 0].axvline(data['session_load'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {data['session_load'].mean():.1f}")
        axes[2, 0].axvline(data['session_load'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {data['session_load'].median():.1f}")
        axes[2, 0].set_title('Load Distribution', fontweight='bold')
        axes[2, 0].set_xlabel('Session Load (AU)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3, axis='y')
        
        # Plot 6: Weekly Aggregation
        data_weekly = data.copy()
        data_weekly['week'] = data_weekly['date'].dt.isocalendar().week
        weekly_summary = data_weekly.groupby('week')['session_load'].sum()
        
        if len(weekly_summary) > 0:
            axes[2, 1].bar(range(len(weekly_summary)), weekly_summary.values, alpha=0.8, edgecolor='black', color='#27ae60')
            axes[2, 1].set_title('Weekly Total Load', fontweight='bold')
            axes[2, 1].set_xlabel('Week Number')
            axes[2, 1].set_ylabel('Total Load (AU)')
            axes[2, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_file}")
        
        return fig
    
    @staticmethod
    def create_team_comparison(df: pd.DataFrame, output_file: str = 'team_comparison.png'):
        """Create team-wide comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Team Load Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Total load by athlete
        athlete_loads = df.groupby('athlete_id')['session_load'].sum().sort_values(ascending=False)
        axes[0, 0].barh(range(min(15, len(athlete_loads))), athlete_loads[:15].values, color='#3498db', alpha=0.8)
        axes[0, 0].set_yticks(range(min(15, len(athlete_loads))))
        axes[0, 0].set_yticklabels(athlete_loads[:15].index)
        axes[0, 0].set_title('Top 15 Athletes by Total Load', fontweight='bold')
        axes[0, 0].set_xlabel('Total Load (AU)')
        axes[0, 0].grid(alpha=0.3, axis='x')
        
        # Plot 2: ACWR distribution
        if 'acwr' in df.columns:
            axes[0, 1].hist(df['acwr'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='#9b59b6')
            axes[0, 1].axvline(1.3, color='green', linestyle='--', linewidth=2, label='Optimal (1.3)')
            axes[0, 1].axvline(1.5, color='red', linestyle='--', linewidth=2, label='High Risk (1.5)')
            axes[0, 1].set_title('Team ACWR Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('ACWR')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Plot 3: Average load by session type
        if 'session_type' in df.columns:
            session_avg = df.groupby('session_type')['session_load'].mean().sort_values()
            colors = plt.cm.viridis(np.linspace(0, 1, len(session_avg)))
            axes[1, 0].barh(range(len(session_avg)), session_avg.values, color=colors, alpha=0.8)
            axes[1, 0].set_yticks(range(len(session_avg)))
            axes[1, 0].set_yticklabels(session_avg.index)
            axes[1, 0].set_title('Average Load by Session Type', fontweight='bold')
            axes[1, 0].set_xlabel('Average Load (AU)')
            axes[1, 0].grid(alpha=0.3, axis='x')
        
        # Plot 4: Team daily average
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        daily_avg = df_temp.groupby('date')['session_load'].mean()
        
        axes[1, 1].plot(daily_avg.index, daily_avg.values, linewidth=2, color='#e74c3c', alpha=0.7)
        axes[1, 1].fill_between(daily_avg.index, daily_avg.values, alpha=0.3, color='#e74c3c')
        axes[1, 1].set_title('Team Average Daily Load', fontweight='bold')
        axes[1, 1].set_ylabel('Average Load (AU)')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Team comparison saved: {output_file}")
        
        return fig


if __name__ == "__main__":
    print("=" * 80)
    print("VISUALIZATION DEMO")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    sample_df = pd.DataFrame({
        'athlete_id': ['ATH_001'] * n_days,
        'date': dates,
        'session_load': [max(0, 500 + np.random.normal(0, 100)) for _ in range(n_days)],
        'session_type': np.random.choice(['Training', 'Match', 'Recovery'], n_days),
        'acwr': np.random.uniform(0.8, 1.4, n_days),
        'acute_load': [max(0, 400 + np.random.normal(0, 80)) for _ in range(n_days)],
        'chronic_load': [max(0, 380 + np.random.normal(0, 60)) for _ in range(n_days)],
        'training_monotony': np.random.uniform(1.5, 3.0, n_days)
    })
    
    LoadVisualizer.create_dashboard(sample_df, 'ATH_001', 'sample_dashboard.png')
    LoadVisualizer.create_team_comparison(sample_df, 'sample_team.png')
    
    print("\n✅ COMPLETE")