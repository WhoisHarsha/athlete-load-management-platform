-- ============================================================================
-- INJURY DISTRIBUTION ANALYSIS
-- ============================================================================
-- Purpose: Analyze injury patterns by muscle group, severity, and type
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- ============================================================================

-- Query 1: Injury Distribution by Muscle Group
-- Shows which muscle groups are most frequently injured
SELECT 
    muscle_group_affected,
    COUNT(*) as total_injuries,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM injuries), 2) as percentage,
    ROUND(AVG(injury_severity), 2) as avg_severity,
    MIN(injury_severity) as min_severity,
    MAX(injury_severity) as max_severity
FROM injuries
GROUP BY muscle_group_affected
ORDER BY total_injuries DESC;

-- Query 2: Injury Distribution by Severity Levels
-- Categorizes injuries into Minor, Moderate, and Severe
SELECT 
    CASE 
        WHEN injury_severity BETWEEN 1 AND 3 THEN 'Minor (1-3)'
        WHEN injury_severity BETWEEN 4 AND 6 THEN 'Moderate (4-6)'
        WHEN injury_severity BETWEEN 7 AND 10 THEN 'Severe (7-10)'
    END as severity_category,
    COUNT(*) as injury_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM injuries), 2) as percentage,
    ROUND(AVG(expected_recovery_days), 1) as avg_expected_recovery_days,
    ROUND(AVG(COALESCE(actual_recovery_days, expected_recovery_days)), 1) as avg_actual_recovery_days
FROM injuries
GROUP BY severity_category
ORDER BY 
    CASE severity_category
        WHEN 'Minor (1-3)' THEN 1
        WHEN 'Moderate (4-6)' THEN 2
        WHEN 'Severe (7-10)' THEN 3
    END;

-- Query 3: Injury Type Distribution with Recovery Statistics
-- Analyzes each injury type with recovery outcomes
SELECT 
    injury_type,
    COUNT(*) as total_injuries,
    ROUND(AVG(injury_severity), 2) as avg_severity,
    ROUND(AVG(expected_recovery_days), 1) as avg_expected_recovery,
    ROUND(AVG(COALESCE(actual_recovery_days, expected_recovery_days)), 1) as avg_actual_recovery,
    COUNT(CASE WHEN requires_surgery THEN 1 END) as surgeries_required,
    COUNT(CASE WHEN injury_status = 'Healed' THEN 1 END) as healed_count,
    COUNT(CASE WHEN injury_status = 'Chronic' THEN 1 END) as chronic_count
FROM injuries
GROUP BY injury_type
ORDER BY total_injuries DESC;

-- Query 4: Muscle Group vs Injury Type Matrix
-- Shows which injury types affect which muscle groups most
SELECT 
    muscle_group_affected,
    injury_type,
    COUNT(*) as injury_count,
    ROUND(AVG(injury_severity), 2) as avg_severity,
    ROUND(AVG(expected_recovery_days), 1) as avg_recovery_days
FROM injuries
GROUP BY muscle_group_affected, injury_type
HAVING COUNT(*) >= 5  -- Only show combinations with 5+ injuries
ORDER BY muscle_group_affected, injury_count DESC;

-- Query 5: Recurrence Analysis
-- Analyzes injury recurrence patterns
SELECT 
    'Total Recurrent Injuries' as metric,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM injuries), 2) as percentage
FROM injuries
WHERE is_recurrence = TRUE

UNION ALL

SELECT 
    'Average Days Between Recurrences' as metric,
    ROUND(AVG(days_since_previous_injury), 1) as count,
    NULL as percentage
FROM injuries
WHERE is_recurrence = TRUE AND days_since_previous_injury IS NOT NULL

UNION ALL

SELECT 
    'Recurrent Injuries with Higher Severity' as metric,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM injuries WHERE is_recurrence = TRUE), 2) as percentage
FROM injuries
WHERE is_recurrence = TRUE AND injury_severity >= 7;

-- Query 6: Monthly Injury Trends
-- Shows injury patterns over time
SELECT 
    TO_CHAR(injury_date, 'YYYY-MM') as month,
    COUNT(*) as total_injuries,
    ROUND(AVG(injury_severity), 2) as avg_severity,
    COUNT(CASE WHEN is_recurrence THEN 1 END) as recurrences,
    COUNT(CASE WHEN requires_surgery THEN 1 END) as surgeries
FROM injuries
GROUP BY TO_CHAR(injury_date, 'YYYY-MM')
ORDER BY month;

-- Query 7: Session Type When Injured
-- Analyzes when injuries occur (training vs match vs recovery)
SELECT 
    session_type_when_injured,
    COUNT(*) as injury_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM injuries), 2) as percentage,
    ROUND(AVG(injury_severity), 2) as avg_severity
FROM injuries
GROUP BY session_type_when_injured
ORDER BY injury_count DESC;

-- Query 8: Top 10 Athletes with Most Injuries
-- Identifies athletes with highest injury frequency
SELECT 
    i.athlete_id,
    a.first_name || ' ' || a.last_name as athlete_name,
    a.position,
    COUNT(*) as total_injuries,
    COUNT(CASE WHEN i.is_recurrence THEN 1 END) as recurrences,
    ROUND(AVG(i.injury_severity), 2) as avg_severity,
    STRING_AGG(DISTINCT i.muscle_group_affected, ', ') as affected_areas
FROM injuries i
JOIN athletes a ON i.athlete_id = a.athlete_id
GROUP BY i.athlete_id, a.first_name, a.last_name, a.position
ORDER BY total_injuries DESC
LIMIT 10;

-- ============================================================================
-- SUMMARY REPORT
-- ============================================================================
SELECT 
    'Total Injuries' as metric,
    COUNT(*)::TEXT as value
FROM injuries

UNION ALL

SELECT 
    'Most Common Muscle Group',
    muscle_group_affected
FROM (
    SELECT muscle_group_affected, COUNT(*) as cnt
    FROM injuries
    GROUP BY muscle_group_affected
    ORDER BY cnt DESC
    LIMIT 1
) sub

UNION ALL

SELECT 
    'Most Common Injury Type',
    injury_type
FROM (
    SELECT injury_type, COUNT(*) as cnt
    FROM injuries
    GROUP BY injury_type
    ORDER BY cnt DESC
    LIMIT 1
) sub

UNION ALL

SELECT 
    'Average Severity',
    ROUND(AVG(injury_severity), 2)::TEXT
FROM injuries

UNION ALL

SELECT 
    'Recurrence Rate (%)',
    ROUND(COUNT(CASE WHEN is_recurrence THEN 1 END) * 100.0 / COUNT(*), 2)::TEXT
FROM injuries;

-- ============================================================================
-- END OF INJURY DISTRIBUTION ANALYSIS
-- ============================================================================