"""
=============================================================================
MEDICAL DATA GENERATION SCRIPT
=============================================================================
Purpose: Generate synthetic medical/injury data for athlete load management
Author: Harsha Prakash (Health Data Science)
Date: November 2025
Project: Athlete Load Management & Performance Optimization Platform

Output Files:
    - athletes.csv (50 athletes)
    - injuries.csv (~1000-1500 injury records)
    - recovery_metrics.csv (~2000-3000 recovery assessments)
    - risk_predictions.csv (5000 risk predictions)
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("MEDICAL DATA GENERATION - STARTING")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_ATHLETES = 50
NUM_RISK_PREDICTIONS = 5000
START_DATE = datetime(2024, 5, 1)  # 6 months of data
END_DATE = datetime(2024, 11, 17)
DAYS_RANGE = (END_DATE - START_DATE).days

# =============================================================================
# TABLE 1: GENERATE ATHLETES MASTER DATA
# =============================================================================
print("\n[1/4] Generating Athletes Master Data...")

positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper', 'Winger', 'Striker']
dominant_sides = ['Right', 'Left', 'Both']
first_names = ['John', 'Michael', 'David', 'James', 'Robert', 'William', 'Joseph', 'Thomas', 'Christopher', 'Daniel',
               'Sarah', 'Emma', 'Emily', 'Jessica', 'Ashley', 'Olivia', 'Sophia', 'Ava', 'Isabella', 'Mia',
               'Carlos', 'Juan', 'Luis', 'Diego', 'Pedro', 'Andre', 'Marco', 'Paolo', 'Lucas', 'Rafael',
               'Ahmed', 'Mohammed', 'Hassan', 'Ali', 'Omar', 'Yuki', 'Kenji', 'Hiroshi', 'Takeshi', 'Satoshi',
               'Liam', 'Noah', 'Ethan', 'Mason', 'Logan', 'Alexander', 'Benjamin', 'Samuel', 'Henry', 'Jack']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
              'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson', 'White', 'Harris',
              'Silva', 'Santos', 'Fernandez', 'Costa', 'Alves', 'Pereira', 'Rodrigues', 'Oliveira', 'Sousa', 'Lima',
              'Chen', 'Wang', 'Li', 'Zhang', 'Liu', 'Kumar', 'Patel', 'Singh', 'Khan', 'Ali',
              'Mueller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer', 'Wagner', 'Becker', 'Schulz', 'Hoffman']
teams = ['Alpha United', 'Beta FC', 'Gamma Rangers', 'Delta Athletic', 'Epsilon City']

athletes_data = []
for i in range(1, NUM_ATHLETES + 1):
    athlete_id = f"ATH_{str(i).zfill(3)}"
    
    # Personal info
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    age = random.randint(20, 35)
    dob = datetime.now() - timedelta(days=age*365 + random.randint(0, 365))
    gender = random.choice(['Male', 'Female'])
    
    # Athletic profile
    position = random.choice(positions)
    dominant_side = random.choice(dominant_sides)
    years_exp = random.randint(2, age - 18)
    height = np.random.normal(178 if gender == 'Male' else 168, 8)
    weight = np.random.normal(75 if gender == 'Male' else 62, 8)
    
    # Team info
    team = random.choice(teams)
    jersey = random.randint(1, 99)
    contract_start = START_DATE - timedelta(days=random.randint(30, 730))
    contract_end = contract_start + timedelta(days=random.randint(730, 1460))
    
    # Status
    status = random.choice(['Active'] * 7 + ['Injured'] * 2 + ['Recovering'] * 1)
    fitness = random.choice(['Excellent', 'Good', 'Fair'])
    
    athletes_data.append({
        'athlete_id': athlete_id,
        'first_name': first_name,
        'last_name': last_name,
        'date_of_birth': dob.strftime('%Y-%m-%d'),
        'gender': gender,
        'position': position,
        'dominant_side': dominant_side,
        'years_experience': years_exp,
        'height_cm': round(height, 2),
        'weight_kg': round(weight, 2),
        'team_name': team,
        'jersey_number': jersey,
        'contract_start_date': contract_start.strftime('%Y-%m-%d'),
        'contract_end_date': contract_end.strftime('%Y-%m-%d'),
        'current_status': status,
        'fitness_level': fitness,
        'created_at': START_DATE.strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

df_athletes = pd.DataFrame(athletes_data)
print(f"   ✓ Generated {len(df_athletes)} athlete records")

# =============================================================================
# TABLE 2: GENERATE INJURIES DATA
# =============================================================================
print("\n[2/4] Generating Injuries Data...")

injury_types = ['Strain', 'Sprain', 'Tear', 'Inflammation', 'Tendinitis', 'Fracture', 'Contusion']
muscle_groups = ['Hamstring', 'Quadriceps', 'Calf', 'Groin', 'Lower Back', 'Shoulder', 'Knee', 'Ankle', 'Hip Flexor', 'Achilles']
session_types = ['Training', 'Match', 'Recovery', 'Practice']
injury_statuses = ['Active', 'Recovering', 'Healed', 'Chronic']
mechanisms = ['Non-contact sprint', 'Collision', 'Overuse', 'Landing', 'Change of direction', 'Tackling', 'Sudden acceleration']

injuries_data = []
injury_count = 0

# Generate 20-40 injuries per athlete over 6 months
for athlete_id in df_athletes['athlete_id']:
    num_injuries = random.randint(15, 35)  # Will result in ~1000-1500 total injuries
    
    athlete_injury_dates = []
    
    for _ in range(num_injuries):
        injury_count += 1
        injury_id = f"INJ_{str(injury_count).zfill(5)}"
        
        # Injury date (distributed over 6 months)
        injury_date = START_DATE + timedelta(days=random.randint(0, DAYS_RANGE))
        
        # Check for recurrence (if athlete had injury in same muscle group within 90 days)
        is_recurrence = False
        prev_injury_id = None
        days_since_prev = None
        
        for prev_date, prev_muscle, prev_id in athlete_injury_dates:
            days_diff = (injury_date - prev_date).days
            if 0 < days_diff <= 90 and random.random() < 0.3:  # 30% chance of recurrence
                is_recurrence = True
                prev_injury_id = prev_id
                days_since_prev = days_diff
                muscle_group = prev_muscle
                break
        
        if not is_recurrence:
            muscle_group = random.choice(muscle_groups)
        
        injury_type = random.choice(injury_types)
        
        # Severity based on type
        if injury_type in ['Tear', 'Fracture']:
            severity = random.randint(6, 10)
        elif injury_type in ['Strain', 'Sprain']:
            severity = random.randint(3, 7)
        else:
            severity = random.randint(2, 6)
        
        # Increase severity if recurrence
        if is_recurrence:
            severity = min(10, severity + random.randint(1, 2))
        
        mechanism = random.choice(mechanisms)
        session_type = random.choice(session_types)
        diagnosis_date = injury_date + timedelta(days=random.randint(0, 2))
        
        requires_surgery = severity >= 8 and random.random() < 0.4
        
        # Status based on severity and time
        if severity <= 3:
            status = random.choice(['Healed', 'Recovering'])
        elif severity <= 6:
            status = random.choice(['Recovering', 'Healed', 'Active'])
        else:
            status = random.choice(['Active', 'Recovering', 'Chronic'])
        
        # Recovery timeline
        expected_recovery = severity * random.randint(4, 8)
        
        if status == 'Healed':
            actual_recovery = expected_recovery + random.randint(-5, 10)
            return_date = injury_date + timedelta(days=actual_recovery)
        elif status == 'Recovering':
            actual_recovery = None
            return_date = None
        else:
            actual_recovery = None
            return_date = None
        
        injuries_data.append({
            'injury_id': injury_id,
            'athlete_id': athlete_id,
            'injury_date': injury_date.strftime('%Y-%m-%d'),
            'injury_type': injury_type,
            'muscle_group_affected': muscle_group,
            'injury_severity': severity,
            'mechanism_of_injury': mechanism,
            'session_type_when_injured': session_type,
            'diagnosis_date': diagnosis_date.strftime('%Y-%m-%d'),
            'diagnosed_by': f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Garcia'])}",
            'medical_notes': f"{injury_type} of {muscle_group} - Grade {severity}/10",
            'requires_surgery': requires_surgery,
            'injury_status': status,
            'expected_recovery_days': expected_recovery,
            'actual_recovery_days': actual_recovery,
            'return_to_play_date': return_date.strftime('%Y-%m-%d') if return_date else None,
            'is_recurrence': is_recurrence,
            'previous_injury_id': prev_injury_id,
            'days_since_previous_injury': days_since_prev,
            'created_at': injury_date.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Track injury for recurrence checking
        athlete_injury_dates.append((injury_date, muscle_group, injury_id))

df_injuries = pd.DataFrame(injuries_data)
print(f"   ✓ Generated {len(df_injuries)} injury records")
print(f"   ✓ Recurrence rate: {df_injuries['is_recurrence'].sum() / len(df_injuries) * 100:.1f}%")

# =============================================================================
# TABLE 3: GENERATE RECOVERY METRICS DATA
# =============================================================================
print("\n[3/4] Generating Recovery Metrics Data...")

recovery_data = []
recovery_count = 0
treatment_types = ['Physiotherapy', 'Ice/Heat Therapy', 'Massage', 'Medication', 'Rest', 'Stretching', 'Strengthening']
progress_statuses = ['Excellent', 'On Track', 'Slow Progress', 'Setback', 'Plateaued']

# Generate recovery assessments for each injury
for _, injury in df_injuries.iterrows():
    injury_id = injury['injury_id']
    athlete_id = injury['athlete_id']
    injury_date = datetime.strptime(injury['injury_date'], '%Y-%m-%d')
    severity = injury['injury_severity']
    status = injury['injury_status']
    
    # Number of recovery assessments based on severity (weekly assessments)
    if severity <= 3:
        num_assessments = random.randint(1, 3)
    elif severity <= 6:
        num_assessments = random.randint(3, 6)
    else:
        num_assessments = random.randint(6, 12)
    
    # Skip if healed quickly
    if status == 'Healed' and severity <= 3:
        num_assessments = random.randint(1, 2)
    
    for week in range(1, num_assessments + 1):
        recovery_count += 1
        recovery_id = f"REC_{str(recovery_count).zfill(6)}"
        
        assessment_date = injury_date + timedelta(days=week * 7)
        
        # Recovery progression (increases over time)
        base_recovery = (week / num_assessments) * 100
        recovery_pct = min(100, base_recovery + random.uniform(-10, 15))
        
        # Pain decreases over time
        pain = max(0, 10 - (week / num_assessments) * 10 + random.uniform(-2, 2))
        
        # Range of motion and strength improve over time
        rom = min(100, recovery_pct + random.uniform(-10, 5))
        strength = min(100, recovery_pct * 0.9 + random.uniform(-10, 5))
        
        # Functional tests (improve over time)
        can_jog = week >= 2 and recovery_pct >= 40
        can_sprint = week >= 4 and recovery_pct >= 60
        can_change_dir = week >= 5 and recovery_pct >= 70
        can_jump = week >= 5 and recovery_pct >= 70
        
        # Treatment
        treatment = random.choice(treatment_types)
        frequency = random.choice(['Daily', '3x per week', '2x per week', 'As needed'])
        
        # Medical clearance
        cleared_training = recovery_pct >= 70 and pain <= 3
        cleared_comp = recovery_pct >= 90 and pain <= 1
        
        # Progress status
        if recovery_pct >= 90:
            prog_status = 'Excellent'
        elif recovery_pct >= base_recovery:
            prog_status = 'On Track'
        elif recovery_pct >= base_recovery - 15:
            prog_status = 'Slow Progress'
        else:
            prog_status = random.choice(['Setback', 'Plateaued'])
        
        restrictions = None
        if not cleared_comp and cleared_training:
            restrictions = random.choice(['No contact drills', 'Limited intensity', 'Modified exercises'])
        
        milestone = None
        if week == 2:
            milestone = 'Full weight bearing'
        elif week == 4 and can_jog:
            milestone = 'Return to jogging'
        elif week == 6 and can_sprint:
            milestone = 'Return to sprinting'
        elif recovery_pct >= 90:
            milestone = 'Cleared for competition'
        
        recovery_data.append({
            'recovery_id': recovery_id,
            'injury_id': injury_id,
            'athlete_id': athlete_id,
            'assessment_date': assessment_date.strftime('%Y-%m-%d'),
            'assessment_week': week,
            'recovery_percentage': round(recovery_pct, 2),
            'pain_level': round(pain, 1),
            'range_of_motion_percentage': round(rom, 2),
            'strength_percentage': round(strength, 2),
            'can_jog': can_jog,
            'can_sprint': can_sprint,
            'can_change_direction': can_change_dir,
            'can_jump': can_jump,
            'treatment_type': treatment,
            'treatment_frequency': frequency,
            'rehabilitation_exercises': f"Week {week} protocol for {injury['muscle_group_affected']}",
            'cleared_for_training': cleared_training,
            'cleared_for_competition': cleared_comp,
            'restrictions': restrictions,
            'progress_status': prog_status,
            'therapist_name': f"PT {random.choice(['Anderson', 'Brown', 'Davis', 'Evans', 'Foster'])}",
            'assessment_notes': f"Week {week} assessment - {prog_status}",
            'next_assessment_date': (assessment_date + timedelta(days=7)).strftime('%Y-%m-%d') if week < num_assessments else None,
            'milestone_reached': milestone,
            'created_at': assessment_date.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

df_recovery = pd.DataFrame(recovery_data)
print(f"   ✓ Generated {len(df_recovery)} recovery assessment records")

# =============================================================================
# TABLE 4: GENERATE RISK PREDICTIONS DATA
# =============================================================================
print("\n[4/4] Generating Risk Predictions Data...")

prediction_periods = ['Next 7 Days', 'Next 14 Days', 'Next 30 Days']
model_versions = ['RandomForest_v1.2', 'XGBoost_v2.0', 'RandomForest_v1.3']

predictions_data = []

# Generate ~100 predictions per athlete (5000 total)
predictions_per_athlete = NUM_RISK_PREDICTIONS // NUM_ATHLETES

for idx, athlete_id in enumerate(df_athletes['athlete_id']):
    # Get athlete's injury history
    athlete_injuries = df_injuries[df_injuries['athlete_id'] == athlete_id]
    injury_count = len(athlete_injuries)
    
    for i in range(predictions_per_athlete):
        pred_id = f"PRED_{str(idx * predictions_per_athlete + i + 1).zfill(6)}"
        
        # Prediction date
        pred_date = START_DATE + timedelta(days=random.randint(0, DAYS_RANGE))
        pred_period = random.choice(prediction_periods)
        
        # Calculate risk factors
        acwr = np.random.normal(1.0, 0.3)
        acwr = max(0.5, min(2.0, acwr))
        
        workload_spike = acwr > 1.5
        fatigue = np.random.normal(5, 2)
        fatigue = max(0, min(10, fatigue))
        
        sleep_quality = np.random.normal(7, 1.5)
        sleep_quality = max(0, min(10, sleep_quality))
        
        soreness = np.random.normal(4, 2)
        soreness = max(0, min(10, soreness))
        
        monotony = np.random.normal(2, 0.5)
        monotony = max(1, min(4, monotony))
        
        # Days since last injury
        if injury_count > 0:
            recent_injuries = athlete_injuries[
                pd.to_datetime(athlete_injuries['injury_date']) <= pred_date
            ]
            if len(recent_injuries) > 0:
                last_injury_date = pd.to_datetime(recent_injuries['injury_date']).max()
                days_since = (pred_date - last_injury_date).days
                prev_severity_avg = recent_injuries['injury_severity'].mean()
            else:
                days_since = 999
                prev_severity_avg = 0
        else:
            days_since = 999
            prev_severity_avg = 0
        
        # Calculate risk score
        risk_score = (
            (acwr - 0.8) * 20 +
            fatigue * 4 +
            (10 - sleep_quality) * 3 +
            soreness * 2.5 +
            monotony * 6 +
            (injury_count * 3) +
            (max(0, 90 - days_since) / 90 * 25) +
            (prev_severity_avg * 2) +
            random.uniform(-5, 5)
        )
        risk_score = max(0, min(100, risk_score))
        
        # Recurrence multiplier
        recurrence_mult = 1.0 + (injury_count * 0.1)
        
        # Predicted injury details (if high risk)
        if risk_score > 65:
            likely_injury = random.choice(injury_types)
            likely_muscle = random.choice(muscle_groups)
        else:
            likely_injury = None
            likely_muscle = None
        
        confidence = np.random.normal(75, 10)
        confidence = max(50, min(95, confidence))
        
        # Actual outcome (simulate)
        injury_probability = risk_score / 100 * 0.3  # High risk ~30% chance
        actual_injury = random.random() < injury_probability
        
        # Find actual injury if it occurred
        actual_injury_id = None
        if actual_injury:
            # Check if there's an injury within prediction period
            period_days = int(pred_period.split()[1])
            period_end = pred_date + timedelta(days=period_days)
            
            period_injuries = athlete_injuries[
                (pd.to_datetime(athlete_injuries['injury_date']) >= pred_date) &
                (pd.to_datetime(athlete_injuries['injury_date']) <= period_end)
            ]
            
            if len(period_injuries) > 0:
                actual_injury_id = period_injuries.iloc[0]['injury_id']
            else:
                actual_injury = False
        
        outcome_date = pred_date + timedelta(days=int(pred_period.split()[1]))
        
        # Recommendations
        if risk_score > 65:
            recommendations = random.choice([
                'Reduce training load by 20-30%',
                'Increase recovery time between sessions',
                'Focus on preventive strengthening exercises',
                'Monitor closely for next 7 days',
                'Implement load management protocol'
            ])
        else:
            recommendations = 'Continue current training program'
        
        intervention = risk_score > 75
        intervention_desc = recommendations if intervention else None
        
        # Top features
        features = [
            ('ACWR', abs(acwr - 1.0) * 50),
            ('Fatigue', fatigue * 10),
            ('Sleep', (10 - sleep_quality) * 10),
            ('Injury History', injury_count * 15),
            ('Days Since Injury', max(0, 90 - days_since)),
            ('Soreness', soreness * 8)
        ]
        features.sort(key=lambda x: x[1], reverse=True)
        top3_features = ', '.join([f[0] for f in features[:3]])
        
        predictions_data.append({
            'prediction_id': pred_id,
            'athlete_id': athlete_id,
            'prediction_date': pred_date.strftime('%Y-%m-%d'),
            'prediction_period': pred_period,
            'predicted_risk_score': round(risk_score, 2),
            'acute_chronic_workload_ratio': round(acwr, 2),
            'recent_workload_spike': workload_spike,
            'fatigue_score': round(fatigue, 2),
            'sleep_quality_score': round(sleep_quality, 2),
            'muscle_soreness_level': round(soreness, 2),
            'training_monotony': round(monotony, 2),
            'injury_history_count': injury_count,
            'days_since_last_injury': days_since,
            'previous_injury_severity_avg': round(prev_severity_avg, 2) if prev_severity_avg > 0 else None,
            'recurrence_risk_multiplier': round(recurrence_mult, 2),
            'likely_injury_type': likely_injury,
            'likely_muscle_group': likely_muscle,
            'confidence_level': round(confidence, 2),
            'actual_injury_occurred': actual_injury,
            'actual_injury_id': actual_injury_id,
            'outcome_recorded_date': outcome_date.strftime('%Y-%m-%d'),
            'recommended_actions': recommendations,
            'intervention_applied': intervention,
            'intervention_description': intervention_desc,
            'model_version': random.choice(model_versions),
            'feature_importance_top3': top3_features,
            'created_at': pred_date.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

df_predictions = pd.DataFrame(predictions_data)
print(f"   ✓ Generated {len(df_predictions)} risk prediction records")

# =============================================================================
# SAVE ALL DATASETS
# =============================================================================
print("\n" + "=" * 80)
print("SAVING DATA TO CSV FILES")
print("=" * 80)

df_athletes.to_csv('athletes.csv', index=False)
print(f"✓ Saved: athletes.csv ({len(df_athletes)} rows)")

df_injuries.to_csv('injuries.csv', index=False)
print(f"✓ Saved: injuries.csv ({len(df_injuries)} rows)")

df_recovery.to_csv('recovery_metrics.csv', index=False)
print(f"✓ Saved: recovery_metrics.csv ({len(df_recovery)} rows)")

df_predictions.to_csv('risk_predictions.csv', index=False)
print(f"✓ Saved: risk_predictions.csv ({len(df_predictions)} rows)")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DATA GENERATION SUMMARY")
print("=" * 80)

print(f"\n📊 ATHLETES TABLE:")
print(f"   Total Athletes: {len(df_athletes)}")
print(f"   Positions: {df_athletes['position'].value_counts().to_dict()}")
print(f"   Status Distribution: {df_athletes['current_status'].value_counts().to_dict()}")

print(f"\n🏥 INJURIES TABLE:")
print(f"   Total Injuries: {len(df_injuries)}")
print(f"   Avg Injuries per Athlete: {len(df_injuries) / NUM_ATHLETES:.1f}")
print(f"   Severity Distribution:")
print(f"      Minor (1-3): {len(df_injuries[df_injuries['injury_severity'] <= 3])}")
print(f"      Moderate (4-6): {len(df_injuries[(df_injuries['injury_severity'] >= 4) & (df_injuries['injury_severity'] <= 6)])}")
print(f"      Severe (7-10): {len(df_injuries[df_injuries['injury_severity'] >= 7])}")
print(f"   Most Common Muscle Groups:")
for muscle, count in df_injuries['muscle_group_affected'].value_counts().head(3).items():
    print(f"      {muscle}: {count}")

print(f"\n💪 RECOVERY METRICS TABLE:")
print(f"   Total Assessments: {len(df_recovery)}")
print(f"   Avg Assessments per Injury: {len(df_recovery) / len(df_injuries):.1f}")
print(f"   Progress Status Distribution: {df_recovery['progress_status'].value_counts().to_dict()}")

print(f"\n⚠️  RISK PREDICTIONS TABLE:")
print(f"   Total Predictions: {len(df_predictions)}")
print(f"   Risk Categories:")
print(f"      Low Risk (0-30): {len(df_predictions[df_predictions['predicted_risk_score'] <= 30])}")
print(f"      Moderate Risk (31-65): {len(df_predictions[(df_predictions['predicted_risk_score'] > 30) & (df_predictions['predicted_risk_score'] <= 65)])}")
print(f"      High Risk (66-100): {len(df_predictions[df_predictions['predicted_risk_score'] > 65])}")
print(f"   Actual Injury Rate: {df_predictions['actual_injury_occurred'].sum() / len(df_predictions) * 100:.1f}%")
print(f"   High Risk Predictions: {len(df_predictions[df_predictions['predicted_risk_score'] > 65])}")
print(f"   Interventions Applied: {df_predictions['intervention_applied'].sum()}")

print("\n" + "=" * 80)
print("✅ DATA GENERATION COMPLETE!")
print("=" * 80)
print("\nNext Steps:")
print("1. Load data into PostgreSQL database")
print("2. Run SQL analysis queries")
print("3. Build injury prediction ML model")
print("=" * 80)