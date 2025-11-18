-- ============================================================================
-- RECOVERY PROGRESS TRACKING ANALYSIS
-- ============================================================================
-- Purpose: Track recovery timelines, milestones, and progress patterns
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- ============================================================================

-- Query 1: Average Recovery Timeline by Severity
-- Shows how recovery progresses across different severity levels
SELECT 
    CASE 
        WHEN i.injury_severity BETWEEN 1 AND 3 THEN 'Minor (1-3)'
        WHEN i.injury_severity BETWEEN 4 AND 6 THEN 'Moderate (4-6)'
        WHEN i.injury_severity BETWEEN 7 AND 10 THEN 'Severe (7-10)'
    END as severity_category,
    COUNT(DISTINCT rm.injury_id) as injuries_tracked,
    ROUND(AVG(rm.assessment_week), 2) as avg_weeks_in_recovery,
    MAX(rm.assessment_week) as max_weeks_observed,
    ROUND(AVG(rm.recovery_percentage), 2) as avg_recovery_progress,
    ROUND(AVG(rm.pain_level), 2) as avg_pain_level
FROM recovery_metrics rm
JOIN injuries i ON rm.injury_id = i.injury_id
GROUP BY severity_category
ORDER BY 
    CASE severity_category
        WHEN 'Minor (1-3)' THEN 1
        WHEN 'Moderate (4-6)' THEN 2
        WHEN 'Severe (7-10)' THEN 3
    END;

-- Query 2: Week-by-Week Recovery Progression
-- Tracks average recovery metrics across weeks
SELECT 
    assessment_week,
    COUNT(*) as total_assessments,
    ROUND(AVG(recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(pain_level), 2) as avg_pain_level,
    ROUND(AVG(range_of_motion_percentage), 2) as avg_rom_pct,
    ROUND(AVG(strength_percentage), 2) as avg_strength_pct,
    COUNT(CASE WHEN cleared_for_training THEN 1 END) as cleared_training_count,
    COUNT(CASE WHEN cleared_for_competition THEN 1 END) as cleared_competition_count
FROM recovery_metrics
WHERE assessment_week <= 12  -- Focus on first 12 weeks
GROUP BY assessment_week
ORDER BY assessment_week;

-- Query 3: Recovery Progress Status Distribution
-- Shows how many injuries are progressing well vs having setbacks
SELECT 
    progress_status,
    COUNT(*) as assessment_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM recovery_metrics), 2) as percentage,
    ROUND(AVG(recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(pain_level), 2) as avg_pain_level
FROM recovery_metrics
GROUP BY progress_status
ORDER BY assessment_count DESC;

-- Query 4: Milestone Achievement Analysis
-- Tracks when athletes reach key recovery milestones
SELECT 
    milestone_reached,
    COUNT(*) as times_reached,
    ROUND(AVG(assessment_week), 2) as avg_week_reached,
    ROUND(AVG(recovery_percentage), 2) as avg_recovery_at_milestone,
    ROUND(AVG(pain_level), 2) as avg_pain_at_milestone
FROM recovery_metrics
WHERE milestone_reached IS NOT NULL
GROUP BY milestone_reached
ORDER BY times_reached DESC;

-- Query 5: Treatment Type Effectiveness
-- Analyzes which treatments correlate with better recovery progress
SELECT 
    treatment_type,
    COUNT(*) as treatment_sessions,
    ROUND(AVG(recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(pain_level), 2) as avg_pain_level,
    COUNT(CASE WHEN progress_status = 'Excellent' THEN 1 END) as excellent_progress_count,
    COUNT(CASE WHEN progress_status IN ('Setback', 'Plateaued') THEN 1 END) as poor_progress_count
FROM recovery_metrics
GROUP BY treatment_type
ORDER BY avg_recovery_pct DESC;

-- Query 6: Functional Test Progression
-- Tracks when athletes regain functional abilities
SELECT 
    assessment_week,
    COUNT(*) as total_assessments,
    ROUND(COUNT(CASE WHEN can_jog THEN 1 END) * 100.0 / COUNT(*), 2) as pct_can_jog,
    ROUND(COUNT(CASE WHEN can_sprint THEN 1 END) * 100.0 / COUNT(*), 2) as pct_can_sprint,
    ROUND(COUNT(CASE WHEN can_change_direction THEN 1 END) * 100.0 / COUNT(*), 2) as pct_can_change_dir,
    ROUND(COUNT(CASE WHEN can_jump THEN 1 END) * 100.0 / COUNT(*), 2) as pct_can_jump
FROM recovery_metrics
WHERE assessment_week <= 10
GROUP BY assessment_week
ORDER BY assessment_week;

-- Query 7: Recovery Timeline by Muscle Group
-- Shows which muscle groups take longest to recover
SELECT 
    i.muscle_group_affected,
    COUNT(DISTINCT rm.injury_id) as injuries_tracked,
    ROUND(AVG(rm.assessment_week), 2) as avg_weeks_tracked,
    ROUND(AVG(rm.recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(rm.pain_level), 2) as avg_pain_level,
    COUNT(CASE WHEN rm.progress_status IN ('Setback', 'Plateaued') THEN 1 END) as setback_count
FROM recovery_metrics rm
JOIN injuries i ON rm.injury_id = i.injury_id
GROUP BY i.muscle_group_affected
ORDER BY avg_weeks_tracked DESC;

-- Query 8: Athletes with Slow Recovery Progress
-- Identifies athletes who need additional attention
SELECT 
    rm.athlete_id,
    a.first_name || ' ' || a.last_name as athlete_name,
    i.injury_type,
    i.muscle_group_affected,
    i.injury_severity,
    MAX(rm.assessment_week) as weeks_in_recovery,
    ROUND(AVG(rm.recovery_percentage), 2) as avg_recovery_pct,
    COUNT(CASE WHEN rm.progress_status IN ('Setback', 'Plateaued') THEN 1 END) as setbacks
FROM recovery_metrics rm
JOIN athletes a ON rm.athlete_id = a.athlete_id
JOIN injuries i ON rm.injury_id = i.injury_id
GROUP BY rm.athlete_id, a.first_name, a.last_name, i.injury_type, i.muscle_group_affected, i.injury_severity
HAVING AVG(rm.recovery_percentage) < 60 OR COUNT(CASE WHEN rm.progress_status IN ('Setback', 'Plateaued') THEN 1 END) >= 2
ORDER BY avg_recovery_pct ASC
LIMIT 15;

-- Query 9: Medical Clearance Analysis
-- Tracks when athletes get cleared for training and competition
SELECT 
    CASE 
        WHEN cleared_for_competition THEN 'Cleared for Competition'
        WHEN cleared_for_training THEN 'Cleared for Training Only'
        ELSE 'Not Cleared'
    END as clearance_status,
    COUNT(*) as assessment_count,
    ROUND(AVG(assessment_week), 2) as avg_week,
    ROUND(AVG(recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(pain_level), 2) as avg_pain_level,
    ROUND(AVG(strength_percentage), 2) as avg_strength_pct
FROM recovery_metrics
GROUP BY clearance_status
ORDER BY 
    CASE clearance_status
        WHEN 'Cleared for Competition' THEN 1
        WHEN 'Cleared for Training Only' THEN 2
        ELSE 3
    END;

-- Query 10: Recovery Progress by Injury Type
-- Compares recovery patterns across different injury types
SELECT 
    i.injury_type,
    COUNT(DISTINCT rm.injury_id) as injuries,
    ROUND(AVG(rm.recovery_percentage), 2) as avg_recovery_pct,
    ROUND(AVG(rm.assessment_week), 2) as avg_weeks_tracked,
    COUNT(CASE WHEN rm.cleared_for_competition THEN 1 END) as competition_ready_count,
    COUNT(CASE WHEN rm.progress_status = 'Excellent' THEN 1 END) as excellent_progress_count,
    COUNT(CASE WHEN rm.progress_status IN ('Setback', 'Plateaued', 'Slow Progress') THEN 1 END) as poor_progress_count
FROM recovery_metrics rm
JOIN injuries i ON rm.injury_id = i.injury_id
GROUP BY i.injury_type
ORDER BY avg_recovery_pct DESC;

-- Query 11: Detailed Recovery Journey for Specific Injury
-- Example: Track a single injury's complete recovery timeline
-- (Replace 'INJ_00001' with any injury_id to see its journey)
SELECT 
    assessment_date,
    assessment_week,
    recovery_percentage,
    pain_level,
    range_of_motion_percentage,
    strength_percentage,
    can_jog,
    can_sprint,
    progress_status,
    cleared_for_training,
    cleared_for_competition,
    milestone_reached,
    treatment_type
FROM recovery_metrics
WHERE injury_id = 'INJ_00001'
ORDER BY assessment_week;

-- ============================================================================
-- RECOVERY SUMMARY REPORT
-- ============================================================================
SELECT 
    'Total Recovery Assessments' as metric,
    COUNT(*)::TEXT as value
FROM recovery_metrics

UNION ALL

SELECT 
    'Average Recovery Percentage',
    ROUND(AVG(recovery_percentage), 2)::TEXT || '%'
FROM recovery_metrics

UNION ALL

SELECT 
    'Average Pain Level',
    ROUND(AVG(pain_level), 2)::TEXT
FROM recovery_metrics

UNION ALL

SELECT 
    'Athletes Cleared for Competition',
    COUNT(DISTINCT athlete_id)::TEXT
FROM recovery_metrics
WHERE cleared_for_competition = TRUE

UNION ALL

SELECT 
    'Most Common Progress Status',
    progress_status
FROM (
    SELECT progress_status, COUNT(*) as cnt
    FROM recovery_metrics
    GROUP BY progress_status
    ORDER BY cnt DESC
    LIMIT 1
) sub

UNION ALL

SELECT 
    'Average Weeks in Recovery',
    ROUND(AVG(assessment_week), 2)::TEXT
FROM recovery_metrics;

-- ============================================================================
-- END OF RECOVERY PROGRESS TRACKING ANALYSIS
-- ============================================================================