-- ============================================================================
-- PREDICTED VS ACTUAL INJURY OUTCOMES COMPARISON
-- ============================================================================
-- Purpose: Evaluate ML model performance by comparing predictions to outcomes
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- ============================================================================

-- Query 1: Overall Model Accuracy Metrics
-- Calculates key performance metrics for the prediction model
WITH prediction_stats AS (
    SELECT 
        COUNT(*) as total_predictions,
        COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
        COUNT(CASE WHEN predicted_risk_score > 65 THEN 1 END) as high_risk_predictions,
        COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) as true_positives,
        COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = FALSE THEN 1 END) as false_positives,
        COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = TRUE THEN 1 END) as false_negatives,
        COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END) as true_negatives
    FROM risk_predictions
    WHERE actual_injury_occurred IS NOT NULL
)
SELECT 
    total_predictions,
    actual_injuries,
    high_risk_predictions,
    true_positives as TP,
    false_positives as FP,
    false_negatives as FN,
    true_negatives as TN,
    ROUND((true_positives + true_negatives)::DECIMAL / total_predictions * 100, 2) as accuracy_pct,
    ROUND(true_positives::DECIMAL / NULLIF(true_positives + false_positives, 0) * 100, 2) as precision_pct,
    ROUND(true_positives::DECIMAL / NULLIF(true_positives + false_negatives, 0) * 100, 2) as recall_pct,
    ROUND(2.0 * (true_positives::DECIMAL / NULLIF(true_positives + false_positives, 0)) * 
          (true_positives::DECIMAL / NULLIF(true_positives + false_negatives, 0)) / 
          NULLIF((true_positives::DECIMAL / NULLIF(true_positives + false_positives, 0)) + 
          (true_positives::DECIMAL / NULLIF(true_positives + false_negatives, 0)), 0) * 100, 2) as f1_score_pct
FROM prediction_stats;

-- Query 2: Risk Category Accuracy
-- Evaluates model performance across different risk categories
SELECT 
    CASE 
        WHEN predicted_risk_score <= 30 THEN 'Low Risk (0-30)'
        WHEN predicted_risk_score <= 65 THEN 'Moderate Risk (31-65)'
        ELSE 'High Risk (66-100)'
    END as risk_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score,
    ROUND(AVG(confidence_level), 2) as avg_confidence
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY risk_category
ORDER BY 
    CASE risk_category
        WHEN 'Low Risk (0-30)' THEN 1
        WHEN 'Moderate Risk (31-65)' THEN 2
        ELSE 3
    END;

-- Query 3: Model Performance by Prediction Period
-- Compares accuracy across different prediction timeframes
SELECT 
    prediction_period,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
    COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) as correct_high_risk,
    COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END) as correct_low_risk,
    ROUND((COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) + 
           COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END))::DECIMAL / 
           COUNT(*) * 100, 2) as accuracy_pct
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY prediction_period
ORDER BY prediction_period;

-- Query 4: Model Performance by Model Version
-- Compares different model versions
SELECT 
    model_version,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score,
    ROUND(AVG(confidence_level), 2) as avg_confidence,
    COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) as true_positives,
    COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = FALSE THEN 1 END) as false_positives,
    ROUND((COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) + 
           COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END))::DECIMAL / 
           COUNT(*) * 100, 2) as accuracy_pct
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY model_version
ORDER BY accuracy_pct DESC;

-- Query 5: False Positive Analysis
-- Investigates cases where model predicted injury but none occurred
SELECT 
    rp.athlete_id,
    a.first_name || ' ' || a.last_name as athlete_name,
    rp.prediction_date,
    rp.predicted_risk_score,
    rp.acute_chronic_workload_ratio as acwr,
    rp.fatigue_score,
    rp.injury_history_count,
    rp.days_since_last_injury,
    rp.likely_injury_type,
    rp.likely_muscle_group
FROM risk_predictions rp
JOIN athletes a ON rp.athlete_id = a.athlete_id
WHERE predicted_risk_score > 65 
  AND actual_injury_occurred = FALSE
ORDER BY predicted_risk_score DESC
LIMIT 20;

-- Query 6: False Negative Analysis
-- Investigates cases where model missed actual injuries
SELECT 
    rp.athlete_id,
    a.first_name || ' ' || a.last_name as athlete_name,
    rp.prediction_date,
    rp.predicted_risk_score,
    rp.acute_chronic_workload_ratio as acwr,
    rp.fatigue_score,
    rp.injury_history_count,
    rp.days_since_last_injury,
    i.injury_type as actual_injury_type,
    i.muscle_group_affected as actual_muscle_group,
    i.injury_severity as actual_severity
FROM risk_predictions rp
JOIN athletes a ON rp.athlete_id = a.athlete_id
LEFT JOIN injuries i ON rp.actual_injury_id = i.injury_id
WHERE predicted_risk_score <= 65 
  AND actual_injury_occurred = TRUE
ORDER BY i.injury_severity DESC
LIMIT 20;

-- Query 7: Prediction Accuracy by Risk Factor Combinations
-- Analyzes which risk factors contribute most to accurate predictions
SELECT 
    CASE WHEN acute_chronic_workload_ratio > 1.5 THEN 'High ACWR' ELSE 'Normal ACWR' END as acwr_status,
    CASE WHEN fatigue_score >= 7 THEN 'High Fatigue' ELSE 'Normal Fatigue' END as fatigue_status,
    CASE WHEN injury_history_count >= 3 THEN 'High Injury History' ELSE 'Low Injury History' END as history_status,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY acwr_status, fatigue_status, history_status
HAVING COUNT(*) >= 10
ORDER BY injury_rate_pct DESC;

-- Query 8: Feature Importance Analysis
-- Shows which features are most frequently in top 3 important features
SELECT 
    feature,
    COUNT(*) as times_in_top3,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM risk_predictions), 2) as percentage
FROM (
    SELECT 
        UNNEST(STRING_TO_ARRAY(feature_importance_top3, ', ')) as feature
    FROM risk_predictions
    WHERE feature_importance_top3 IS NOT NULL
) features
GROUP BY feature
ORDER BY times_in_top3 DESC;

-- Query 9: Intervention Effectiveness
-- Evaluates whether interventions reduced injury occurrence
SELECT 
    intervention_applied,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries_occurred,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
  AND predicted_risk_score > 65  -- Only look at high-risk predictions
GROUP BY intervention_applied;

-- Query 10: Monthly Prediction Accuracy Trend
-- Tracks how model performance changes over time
SELECT 
    TO_CHAR(prediction_date, 'YYYY-MM') as month,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score,
    ROUND((COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) + 
           COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END))::DECIMAL / 
           COUNT(*) * 100, 2) as accuracy_pct
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY month
ORDER BY month;

-- Query 11: Confidence Level vs Accuracy
-- Checks if higher confidence correlates with better accuracy
SELECT 
    CASE 
        WHEN confidence_level < 60 THEN 'Low Confidence (<60)'
        WHEN confidence_level < 80 THEN 'Medium Confidence (60-80)'
        ELSE 'High Confidence (80+)'
    END as confidence_bucket,
    COUNT(*) as predictions,
    ROUND(AVG(confidence_level), 2) as avg_confidence,
    ROUND((COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END) + 
           COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END))::DECIMAL / 
           COUNT(*) * 100, 2) as accuracy_pct
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY confidence_bucket
ORDER BY 
    CASE confidence_bucket
        WHEN 'Low Confidence (<60)' THEN 1
        WHEN 'Medium Confidence (60-80)' THEN 2
        ELSE 3
    END;

-- Query 12: Predicted vs Actual Injury Type Match
-- When injuries occurred, did model predict correct type?
SELECT 
    rp.likely_injury_type as predicted_type,
    i.injury_type as actual_type,
    COUNT(*) as occurrences,
    ROUND(AVG(rp.predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions rp
JOIN injuries i ON rp.actual_injury_id = i.injury_id
WHERE rp.actual_injury_occurred = TRUE
  AND rp.likely_injury_type IS NOT NULL
GROUP BY predicted_type, actual_type
HAVING COUNT(*) >= 2
ORDER BY occurrences DESC;

-- ============================================================================
-- CONFUSION MATRIX
-- ============================================================================
SELECT 
    'Confusion Matrix' as metric,
    'Predicted Positive (High Risk)' as predicted_positive,
    'Predicted Negative (Low Risk)' as predicted_negative
UNION ALL
SELECT 
    'Actual Positive (Injury)' as metric,
    COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN 1 END)::TEXT as tp,
    COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = TRUE THEN 1 END)::TEXT as fn
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
UNION ALL
SELECT 
    'Actual Negative (No Injury)' as metric,
    COUNT(CASE WHEN predicted_risk_score > 65 AND actual_injury_occurred = FALSE THEN 1 END)::TEXT as fp,
    COUNT(CASE WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN 1 END)::TEXT as tn
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL;

-- ============================================================================
-- END OF PREDICTED VS ACTUAL COMPARISON
-- ============================================================================