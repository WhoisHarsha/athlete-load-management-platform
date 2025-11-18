-- ============================================================================
-- RISK FACTOR ANALYSIS
-- ============================================================================
-- Purpose: Analyze risk factors and their correlation with injury outcomes
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- ============================================================================

-- Query 1: ACWR (Acute:Chronic Workload Ratio) Impact Analysis
-- Evaluates injury risk at different ACWR levels
SELECT 
    CASE 
        WHEN acute_chronic_workload_ratio < 0.8 THEN 'Very Low (<0.8)'
        WHEN acute_chronic_workload_ratio < 1.0 THEN 'Low (0.8-1.0)'
        WHEN acute_chronic_workload_ratio < 1.3 THEN 'Optimal (1.0-1.3)'
        WHEN acute_chronic_workload_ratio < 1.5 THEN 'Moderate Risk (1.3-1.5)'
        ELSE 'High Risk (>1.5)'
    END as acwr_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(acute_chronic_workload_ratio), 3) as avg_acwr,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY acwr_category
ORDER BY 
    CASE acwr_category
        WHEN 'Very Low (<0.8)' THEN 1
        WHEN 'Low (0.8-1.0)' THEN 2
        WHEN 'Optimal (1.0-1.3)' THEN 3
        WHEN 'Moderate Risk (1.3-1.5)' THEN 4
        ELSE 5
    END;

-- Query 2: Fatigue Score Impact
-- Shows correlation between fatigue and injury occurrence
SELECT 
    CASE 
        WHEN fatigue_score < 4 THEN 'Low Fatigue (0-4)'
        WHEN fatigue_score < 7 THEN 'Moderate Fatigue (4-7)'
        ELSE 'High Fatigue (7-10)'
    END as fatigue_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(fatigue_score), 2) as avg_fatigue,
    ROUND(AVG(sleep_quality_score), 2) as avg_sleep_quality
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY fatigue_category
ORDER BY 
    CASE fatigue_category
        WHEN 'Low Fatigue (0-4)' THEN 1
        WHEN 'Moderate Fatigue (4-7)' THEN 2
        ELSE 3
    END;

-- Query 3: Sleep Quality Impact
-- Evaluates relationship between sleep and injury risk
SELECT 
    CASE 
        WHEN sleep_quality_score < 5 THEN 'Poor Sleep (<5)'
        WHEN sleep_quality_score < 7 THEN 'Fair Sleep (5-7)'
        ELSE 'Good Sleep (7-10)'
    END as sleep_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(sleep_quality_score), 2) as avg_sleep_quality,
    ROUND(AVG(fatigue_score), 2) as avg_fatigue
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY sleep_category
ORDER BY 
    CASE sleep_category
        WHEN 'Poor Sleep (<5)' THEN 1
        WHEN 'Fair Sleep (5-7)' THEN 2
        ELSE 3
    END;

-- Query 4: Injury History Impact
-- Shows how past injuries affect future injury risk
SELECT 
    CASE 
        WHEN injury_history_count = 0 THEN 'No History'
        WHEN injury_history_count <= 2 THEN 'Low History (1-2)'
        WHEN injury_history_count <= 4 THEN 'Moderate History (3-4)'
        ELSE 'High History (5+)'
    END as history_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(injury_history_count), 2) as avg_history_count,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY history_category
ORDER BY 
    CASE history_category
        WHEN 'No History' THEN 1
        WHEN 'Low History (1-2)' THEN 2
        WHEN 'Moderate History (3-4)' THEN 3
        ELSE 4
    END;

-- Query 5: Recent Injury Impact
-- Analyzes injury risk based on days since last injury
SELECT 
    CASE 
        WHEN days_since_last_injury >= 180 THEN 'No Recent Injury (180+ days)'
        WHEN days_since_last_injury >= 90 THEN 'Distant Injury (90-180 days)'
        WHEN days_since_last_injury >= 30 THEN 'Recent Injury (30-90 days)'
        ELSE 'Very Recent Injury (<30 days)'
    END as recency_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(days_since_last_injury), 1) as avg_days_since,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL 
  AND days_since_last_injury < 999  -- Exclude those with no injury history
GROUP BY recency_category
ORDER BY 
    CASE recency_category
        WHEN 'Very Recent Injury (<30 days)' THEN 1
        WHEN 'Recent Injury (30-90 days)' THEN 2
        WHEN 'Distant Injury (90-180 days)' THEN 3
        ELSE 4
    END;

-- Query 6: Training Monotony Impact
-- Evaluates injury risk with training variety
SELECT 
    CASE 
        WHEN training_monotony < 1.5 THEN 'High Variety (<1.5)'
        WHEN training_monotony < 2.5 THEN 'Moderate Variety (1.5-2.5)'
        ELSE 'Low Variety (>2.5)'
    END as monotony_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(training_monotony), 2) as avg_monotony,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY monotony_category
ORDER BY 
    CASE monotony_category
        WHEN 'High Variety (<1.5)' THEN 1
        WHEN 'Moderate Variety (1.5-2.5)' THEN 2
        ELSE 3
    END;

-- Query 7: Muscle Soreness Impact
-- Analyzes correlation between soreness and injury
SELECT 
    CASE 
        WHEN muscle_soreness_level < 4 THEN 'Low Soreness (0-4)'
        WHEN muscle_soreness_level < 7 THEN 'Moderate Soreness (4-7)'
        ELSE 'High Soreness (7-10)'
    END as soreness_category,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(muscle_soreness_level), 2) as avg_soreness,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY soreness_category
ORDER BY 
    CASE soreness_category
        WHEN 'Low Soreness (0-4)' THEN 1
        WHEN 'Moderate Soreness (4-7)' THEN 2
        ELSE 3
    END;

-- Query 8: Workload Spike Analysis
-- Compares injury rates with and without workload spikes
SELECT 
    recent_workload_spike,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(acute_chronic_workload_ratio), 3) as avg_acwr,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY recent_workload_spike;

-- Query 9: Combined Risk Factor Analysis
-- Multiple risk factors present
SELECT 
    CASE 
        WHEN acute_chronic_workload_ratio > 1.5 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN fatigue_score >= 7 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN sleep_quality_score < 6 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN muscle_soreness_level >= 7 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN injury_history_count >= 3 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN days_since_last_injury < 30 THEN 1 ELSE 0 
    END as num_risk_factors,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(predicted_risk_score), 2) as avg_risk_score
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL
GROUP BY num_risk_factors
ORDER BY num_risk_factors;

-- Query 10: Risk Factors by Athlete Demographics
-- Analyzes risk by position and experience
SELECT 
    a.position,
    COUNT(DISTINCT rp.athlete_id) as athletes,
    COUNT(*) as predictions,
    COUNT(CASE WHEN rp.actual_injury_occurred = TRUE THEN 1 END) as injuries,
    ROUND(COUNT(CASE WHEN rp.actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*), 2) as injury_rate_pct,
    ROUND(AVG(rp.predicted_risk_score), 2) as avg_risk_score,
    ROUND(AVG(rp.acute_chronic_workload_ratio), 3) as avg_acwr,
    ROUND(AVG(rp.fatigue_score), 2) as avg_fatigue
FROM risk_predictions rp
JOIN athletes a ON rp.athlete_id = a.athlete_id
WHERE rp.actual_injury_occurred IS NOT NULL
GROUP BY a.position
ORDER BY injury_rate_pct DESC;

-- Query 11: High-Risk Athletes Profile
-- Identifies athletes consistently flagged as high risk
SELECT 
    rp.athlete_id,
    a.first_name || ' ' || a.last_name as athlete_name,
    a.position,
    a.years_experience,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN rp.predicted_risk_score > 65 THEN 1 END) as high_risk_predictions,
    ROUND(COUNT(CASE WHEN rp.predicted_risk_score > 65 THEN 1 END) * 100.0 / COUNT(*), 2) as high_risk_pct,
    COUNT(CASE WHEN rp.actual_injury_occurred = TRUE THEN 1 END) as actual_injuries,
    ROUND(AVG(rp.predicted_risk_score), 2) as avg_risk_score,
    ROUND(AVG(rp.injury_history_count), 2) as avg_injury_history,
    ROUND(AVG(rp.acute_chronic_workload_ratio), 3) as avg_acwr,
    ROUND(AVG(rp.fatigue_score), 2) as avg_fatigue
FROM risk_predictions rp
JOIN athletes a ON rp.athlete_id = a.athlete_id
WHERE rp.actual_injury_occurred IS NOT NULL
GROUP BY rp.athlete_id, a.first_name, a.last_name, a.position, a.years_experience
HAVING COUNT(CASE WHEN rp.predicted_risk_score > 65 THEN 1 END) >= 10
ORDER BY high_risk_pct DESC, actual_injuries DESC
LIMIT 15;

-- Query 12: Risk Factor Correlation Matrix
-- Shows correlations between different risk factors
SELECT 
    ROUND(CORR(acute_chronic_workload_ratio, fatigue_score), 3) as acwr_fatigue_corr,
    ROUND(CORR(fatigue_score, sleep_quality_score), 3) as fatigue_sleep_corr,
    ROUND(CORR(muscle_soreness_level, fatigue_score), 3) as soreness_fatigue_corr,
    ROUND(CORR(training_monotony, fatigue_score), 3) as monotony_fatigue_corr,
    ROUND(CORR(injury_history_count, predicted_risk_score), 3) as history_risk_corr,
    ROUND(CORR(acute_chronic_workload_ratio, predicted_risk_score), 3) as acwr_risk_corr
FROM risk_predictions
WHERE actual_injury_occurred IS NOT NULL;

-- Query 13: Optimal vs Suboptimal Ranges
-- Compares injury rates in optimal vs suboptimal ranges for each factor
WITH risk_ranges AS (
    SELECT 
        athlete_id,
        CASE 
            WHEN acute_chronic_workload_ratio BETWEEN 0.8 AND 1.3 THEN 'Optimal' 
            ELSE 'Suboptimal' 
        END as acwr_range,
        CASE 
            WHEN fatigue_score < 7 THEN 'Optimal' 
            ELSE 'Suboptimal' 
        END as fatigue_range,
        CASE 
            WHEN sleep_quality_score >= 7 THEN 'Optimal' 
            ELSE 'Suboptimal' 
        END as sleep_range,
        CASE 
            WHEN muscle_soreness_level < 7 THEN 'Optimal' 
            ELSE 'Suboptimal' 
        END as soreness_range,
        actual_injury_occurred
    FROM risk_predictions
    WHERE actual_injury_occurred IS NOT NULL
)
SELECT 
    'ACWR' as factor,
    COUNT(CASE WHEN acwr_range = 'Optimal' AND actual_injury_occurred = TRUE THEN 1 END) as optimal_injuries,
    COUNT(CASE WHEN acwr_range = 'Optimal' THEN 1 END) as optimal_total,
    COUNT(CASE WHEN acwr_range = 'Suboptimal' AND actual_injury_occurred = TRUE THEN 1 END) as suboptimal_injuries,
    COUNT(CASE WHEN acwr_range = 'Suboptimal' THEN 1 END) as suboptimal_total
FROM risk_ranges

UNION ALL

SELECT 
    'Fatigue',
    COUNT(CASE WHEN fatigue_range = 'Optimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN fatigue_range = 'Optimal' THEN 1 END),
    COUNT(CASE WHEN fatigue_range = 'Suboptimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN fatigue_range = 'Suboptimal' THEN 1 END)
FROM risk_ranges

UNION ALL

SELECT 
    'Sleep Quality',
    COUNT(CASE WHEN sleep_range = 'Optimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN sleep_range = 'Optimal' THEN 1 END),
    COUNT(CASE WHEN sleep_range = 'Suboptimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN sleep_range = 'Suboptimal' THEN 1 END)
FROM risk_ranges

UNION ALL

SELECT 
    'Muscle Soreness',
    COUNT(CASE WHEN soreness_range = 'Optimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN soreness_range = 'Optimal' THEN 1 END),
    COUNT(CASE WHEN soreness_range = 'Suboptimal' AND actual_injury_occurred = TRUE THEN 1 END),
    COUNT(CASE WHEN soreness_range = 'Suboptimal' THEN 1 END)
FROM risk_ranges;

-- ============================================================================
-- RISK FACTORS SUMMARY
-- ============================================================================
SELECT 
    'Most Predictive Risk Factor' as insight,
    'ACWR > 1.5' as detail
UNION ALL
SELECT 
    'Average Injury Rate with High Risk Factors (3+)',
    ROUND(AVG(injury_rate), 2)::TEXT || '%'
FROM (
    SELECT 
        COUNT(CASE WHEN actual_injury_occurred = TRUE THEN 1 END) * 100.0 / COUNT(*) as injury_rate
    FROM risk_predictions
    WHERE actual_injury_occurred IS NOT NULL
    GROUP BY athlete_id
    HAVING 
        AVG(CASE WHEN acute_chronic_workload_ratio > 1.5 THEN 1 ELSE 0 END) > 0.3 OR
        AVG(CASE WHEN fatigue_score >= 7 THEN 1 ELSE 0 END) > 0.3 OR
        AVG(CASE WHEN sleep_quality_score < 6 THEN 1 ELSE 0 END) > 0.3
) sub;

-- ============================================================================
-- END OF RISK FACTOR ANALYSIS
-- ============================================================================