-- ============================================================================
-- MEDICAL & RECOVERY DATABASE SCHEMA
-- ============================================================================
-- Purpose: Track injuries, recovery metrics, and injury risk predictions
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- Project: Athlete Load Management & Performance Optimization Platform
-- ============================================================================

-- ============================================================================
-- TABLE 1: INJURIES
-- ============================================================================
-- Purpose: Record all injury incidents for athletes
-- Relationship: Many injuries per athlete (1:N with athletes table)

DROP TABLE IF EXISTS injuries CASCADE;

CREATE TABLE injuries (
    -- Primary Key
    injury_id VARCHAR(15) PRIMARY KEY,
    
    -- Foreign Key to athletes table
    athlete_id VARCHAR(10) NOT NULL,
    
    -- Injury Details
    injury_date DATE NOT NULL,
    injury_type VARCHAR(50) NOT NULL
        CHECK (injury_type IN ('Strain', 'Sprain', 'Tear', 'Inflammation', 
                               'Tendinitis', 'Fracture', 'Contusion', 'Other')),
    muscle_group_affected VARCHAR(50) NOT NULL
        CHECK (muscle_group_affected IN ('Hamstring', 'Quadriceps', 'Calf', 
                                          'Groin', 'Lower Back', 'Shoulder', 
                                          'Knee', 'Ankle', 'Hip Flexor', 
                                          'Achilles', 'Other')),
    
    -- Severity Assessment (1-10 scale)
    -- 1-3: Minor, 4-6: Moderate, 7-10: Severe
    injury_severity INT NOT NULL CHECK (injury_severity BETWEEN 1 AND 10),
    
    -- Injury Context
    mechanism_of_injury VARCHAR(100), -- e.g., "Non-contact during sprint", "Collision"
    session_type_when_injured VARCHAR(30), -- e.g., "Training", "Match", "Recovery"
    
    -- Medical Assessment
    diagnosis_date DATE,
    diagnosed_by VARCHAR(100), -- Medical staff name
    medical_notes TEXT,
    requires_surgery BOOLEAN DEFAULT FALSE,
    
    -- Status Tracking
    injury_status VARCHAR(20) DEFAULT 'Active'
        CHECK (injury_status IN ('Active', 'Recovering', 'Healed', 'Chronic', 'Recurrent')),
    
    -- Expected vs Actual Recovery
    expected_recovery_days INT CHECK (expected_recovery_days > 0),
    actual_recovery_days INT CHECK (actual_recovery_days >= 0),
    return_to_play_date DATE,
    
    -- Recurrence Tracking
    is_recurrence BOOLEAN DEFAULT FALSE,
    previous_injury_id VARCHAR(15), -- Links to previous similar injury
    days_since_previous_injury INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraint
    CONSTRAINT fk_injury_athlete 
        FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    
    -- Business Logic Constraints
    CONSTRAINT valid_diagnosis_date 
        CHECK (diagnosis_date >= injury_date),
    CONSTRAINT valid_return_date 
        CHECK (return_to_play_date IS NULL OR return_to_play_date >= injury_date),
    CONSTRAINT valid_actual_recovery 
        CHECK (actual_recovery_days IS NULL OR 
               (return_to_play_date IS NOT NULL AND actual_recovery_days > 0))
);

-- Indexes for performance optimization
CREATE INDEX idx_injuries_athlete ON injuries(athlete_id);
CREATE INDEX idx_injuries_date ON injuries(injury_date DESC);
CREATE INDEX idx_injuries_status ON injuries(injury_status);
CREATE INDEX idx_injuries_severity ON injuries(injury_severity);
CREATE INDEX idx_injuries_muscle_group ON injuries(muscle_group_affected);
CREATE INDEX idx_injuries_type ON injuries(injury_type);

-- Comments for documentation
COMMENT ON TABLE injuries IS 'Records all injury incidents including type, severity, and recovery timeline';
COMMENT ON COLUMN injuries.injury_severity IS 'Severity scale: 1-3 Minor, 4-6 Moderate, 7-10 Severe';
COMMENT ON COLUMN injuries.mechanism_of_injury IS 'Description of how injury occurred (contact vs non-contact)';
COMMENT ON COLUMN injuries.is_recurrence IS 'TRUE if this is a recurrence of a previous injury';
COMMENT ON COLUMN injuries.expected_recovery_days IS 'Initial medical staff estimate for recovery duration';
COMMENT ON COLUMN injuries.actual_recovery_days IS 'Actual days taken from injury to full return to play';


-- ============================================================================
-- TABLE 2: RECOVERY_METRICS
-- ============================================================================
-- Purpose: Track detailed recovery progress with regular assessments
-- Relationship: Many recovery assessments per injury (1:N with injuries table)

DROP TABLE IF EXISTS recovery_metrics CASCADE;

CREATE TABLE recovery_metrics (
    -- Primary Key
    recovery_id VARCHAR(20) PRIMARY KEY,
    
    -- Foreign Keys
    injury_id VARCHAR(15) NOT NULL,
    athlete_id VARCHAR(10) NOT NULL,
    
    -- Assessment Details
    assessment_date DATE NOT NULL,
    assessment_week INT CHECK (assessment_week > 0), -- Week number in recovery timeline
    
    -- Recovery Progress (0-100 scale)
    recovery_percentage DECIMAL(5,2) CHECK (recovery_percentage BETWEEN 0 AND 100),
    pain_level INT CHECK (pain_level BETWEEN 0 AND 10), -- 0 = No pain, 10 = Severe pain
    range_of_motion_percentage DECIMAL(5,2) CHECK (range_of_motion_percentage BETWEEN 0 AND 100),
    strength_percentage DECIMAL(5,2) CHECK (strength_percentage BETWEEN 0 AND 100),
    
    -- Functional Tests
    can_jog BOOLEAN DEFAULT FALSE,
    can_sprint BOOLEAN DEFAULT FALSE,
    can_change_direction BOOLEAN DEFAULT FALSE,
    can_jump BOOLEAN DEFAULT FALSE,
    
    -- Recovery Interventions
    treatment_type VARCHAR(100), -- e.g., "Physiotherapy", "Ice/Heat", "Medication"
    treatment_frequency VARCHAR(50), -- e.g., "Daily", "3x per week"
    rehabilitation_exercises TEXT,
    
    -- Medical Clearance
    cleared_for_training BOOLEAN DEFAULT FALSE,
    cleared_for_competition BOOLEAN DEFAULT FALSE,
    restrictions TEXT, -- Any limitations (e.g., "No contact drills")
    
    -- Progress Status
    progress_status VARCHAR(20) DEFAULT 'On Track'
        CHECK (progress_status IN ('Excellent', 'On Track', 'Slow Progress', 
                                    'Setback', 'Plateaued', 'Regressed')),
    
    -- Assessment Notes
    therapist_name VARCHAR(100),
    assessment_notes TEXT,
    next_assessment_date DATE,
    
    -- Milestones
    milestone_reached VARCHAR(100), -- e.g., "Full weight bearing", "Return to jogging"
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    CONSTRAINT fk_recovery_injury 
        FOREIGN KEY (injury_id) REFERENCES injuries(injury_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_recovery_athlete 
        FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    
    -- Business Logic Constraints
    CONSTRAINT valid_assessment_progression 
        CHECK (next_assessment_date IS NULL OR next_assessment_date > assessment_date),
    CONSTRAINT valid_clearance_logic
        CHECK (NOT cleared_for_competition OR cleared_for_training) -- Can't compete without training clearance
);

-- Indexes
CREATE INDEX idx_recovery_injury ON recovery_metrics(injury_id);
CREATE INDEX idx_recovery_athlete ON recovery_metrics(athlete_id);
CREATE INDEX idx_recovery_date ON recovery_metrics(assessment_date DESC);
CREATE INDEX idx_recovery_status ON recovery_metrics(progress_status);
CREATE INDEX idx_recovery_percentage ON recovery_metrics(recovery_percentage);

-- Comments
COMMENT ON TABLE recovery_metrics IS 'Tracks detailed recovery progress with regular medical assessments';
COMMENT ON COLUMN recovery_metrics.assessment_week IS 'Week number in the recovery timeline (Week 1, Week 2, etc.)';
COMMENT ON COLUMN recovery_metrics.recovery_percentage IS 'Overall recovery progress from 0% (injury) to 100% (full recovery)';
COMMENT ON COLUMN recovery_metrics.pain_level IS 'Current pain level: 0=None, 1-3=Mild, 4-6=Moderate, 7-10=Severe';
COMMENT ON COLUMN recovery_metrics.progress_status IS 'Assessment of recovery trajectory relative to expected timeline';


-- ============================================================================
-- TABLE 3: RISK_PREDICTIONS
-- ============================================================================
-- Purpose: Store injury risk predictions and compare with actual outcomes
-- Relationship: Multiple risk assessments per athlete over time

DROP TABLE IF EXISTS risk_predictions CASCADE;

CREATE TABLE risk_predictions (
    -- Primary Key
    prediction_id VARCHAR(20) PRIMARY KEY,
    
    -- Foreign Key
    athlete_id VARCHAR(10) NOT NULL,
    
    -- Prediction Details
    prediction_date DATE NOT NULL,
    prediction_period VARCHAR(20) NOT NULL, -- e.g., "Next 7 Days", "Next 14 Days", "Next 30 Days"
    
    -- Risk Score (0-100 scale)
    -- 0-30: Low Risk, 31-65: Moderate Risk, 66-100: High Risk
    predicted_risk_score DECIMAL(5,2) NOT NULL CHECK (predicted_risk_score BETWEEN 0 AND 100),
    risk_category VARCHAR(20) GENERATED ALWAYS AS (
        CASE 
            WHEN predicted_risk_score <= 30 THEN 'Low Risk'
            WHEN predicted_risk_score <= 65 THEN 'Moderate Risk'
            ELSE 'High Risk'
        END
    ) STORED,
    
    -- Risk Factors (Contributing factors to the risk score)
    acute_chronic_workload_ratio DECIMAL(4,2), -- ACWR value
    recent_workload_spike BOOLEAN,
    fatigue_score DECIMAL(4,2) CHECK (fatigue_score BETWEEN 0 AND 10),
    sleep_quality_score DECIMAL(4,2) CHECK (sleep_quality_score BETWEEN 0 AND 10),
    muscle_soreness_level DECIMAL(4,2) CHECK (muscle_soreness_level BETWEEN 0 AND 10),
    training_monotony DECIMAL(4,2),
    
    -- Historical Risk Factors
    injury_history_count INT DEFAULT 0 CHECK (injury_history_count >= 0),
    days_since_last_injury INT,
    previous_injury_severity_avg DECIMAL(4,2) CHECK (previous_injury_severity_avg BETWEEN 0 AND 10),
    recurrence_risk_multiplier DECIMAL(3,2) DEFAULT 1.0, -- Increases risk if previous injuries
    
    -- Predicted Injury Details (if risk is high)
    likely_injury_type VARCHAR(50),
    likely_muscle_group VARCHAR(50),
    confidence_level DECIMAL(5,2) CHECK (confidence_level BETWEEN 0 AND 100), -- Model confidence
    
    -- Actual Outcome (filled in after prediction period)
    actual_injury_occurred BOOLEAN, -- TRUE if injury happened during prediction period
    actual_injury_id VARCHAR(15), -- Links to injuries table if injury occurred
    outcome_recorded_date DATE,
    
    -- Model Performance Metrics
    prediction_accuracy BOOLEAN GENERATED ALWAYS AS (
        CASE 
            WHEN actual_injury_occurred IS NULL THEN NULL
            WHEN predicted_risk_score > 65 AND actual_injury_occurred = TRUE THEN TRUE
            WHEN predicted_risk_score <= 65 AND actual_injury_occurred = FALSE THEN TRUE
            ELSE FALSE
        END
    ) STORED,
    
    -- Recommendations
    recommended_actions TEXT, -- e.g., "Reduce training load by 20%", "Increase recovery time"
    intervention_applied BOOLEAN DEFAULT FALSE,
    intervention_description TEXT,
    
    -- Model Information
    model_version VARCHAR(20), -- e.g., "RandomForest_v1.2", "XGBoost_v2.0"
    feature_importance_top3 TEXT, -- Top 3 features that influenced prediction
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    CONSTRAINT fk_prediction_athlete 
        FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_prediction_injury 
        FOREIGN KEY (actual_injury_id) REFERENCES injuries(injury_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    
    -- Business Logic Constraints
    CONSTRAINT valid_outcome_date 
        CHECK (outcome_recorded_date IS NULL OR outcome_recorded_date >= prediction_date)
);

-- Indexes
CREATE INDEX idx_predictions_athlete ON risk_predictions(athlete_id);
CREATE INDEX idx_predictions_date ON risk_predictions(prediction_date DESC);
CREATE INDEX idx_predictions_score ON risk_predictions(predicted_risk_score DESC);
CREATE INDEX idx_predictions_category ON risk_predictions(risk_category);
CREATE INDEX idx_predictions_accuracy ON risk_predictions(prediction_accuracy);
CREATE INDEX idx_predictions_outcome ON risk_predictions(actual_injury_occurred);

-- Comments
COMMENT ON TABLE risk_predictions IS 'Stores ML model predictions for injury risk and compares with actual outcomes';
COMMENT ON COLUMN risk_predictions.predicted_risk_score IS 'Risk score: 0-30 Low, 31-65 Moderate, 66-100 High';
COMMENT ON COLUMN risk_predictions.prediction_period IS 'Time window for prediction (e.g., next 7, 14, or 30 days)';
COMMENT ON COLUMN risk_predictions.prediction_accuracy IS 'Auto-calculated: TRUE if high-risk prediction matched actual outcome';
COMMENT ON COLUMN risk_predictions.confidence_level IS 'ML model confidence in prediction (0-100%)';
COMMENT ON COLUMN risk_predictions.recurrence_risk_multiplier IS 'Multiplier applied to risk score based on injury history';


-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Trigger for injuries table
CREATE OR REPLACE FUNCTION update_injury_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_injury_timestamp
BEFORE UPDATE ON injuries
FOR EACH ROW
EXECUTE FUNCTION update_injury_timestamp();

-- Trigger for recovery_metrics table
CREATE OR REPLACE FUNCTION update_recovery_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_recovery_timestamp
BEFORE UPDATE ON recovery_metrics
FOR EACH ROW
EXECUTE FUNCTION update_recovery_timestamp();

-- Trigger for risk_predictions table
CREATE OR REPLACE FUNCTION update_prediction_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_prediction_timestamp
BEFORE UPDATE ON risk_predictions
FOR EACH ROW
EXECUTE FUNCTION update_prediction_timestamp();


-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Current Active Injuries
CREATE OR REPLACE VIEW active_injuries AS
SELECT 
    i.injury_id,
    i.athlete_id,
    a.first_name,
    a.last_name,
    i.injury_date,
    i.injury_type,
    i.muscle_group_affected,
    i.injury_severity,
    i.injury_status,
    CURRENT_DATE - i.injury_date AS days_injured
FROM injuries i
JOIN athletes a ON i.athlete_id = a.athlete_id
WHERE i.injury_status IN ('Active', 'Recovering');

COMMENT ON VIEW active_injuries IS 'Shows all current active and recovering injuries with athlete details';

-- View: High Risk Athletes
CREATE OR REPLACE VIEW high_risk_athletes AS
SELECT 
    rp.athlete_id,
    a.first_name,
    a.last_name,
    a.position,
    rp.predicted_risk_score,
    rp.risk_category,
    rp.prediction_date,
    rp.recommended_actions
FROM risk_predictions rp
JOIN athletes a ON rp.athlete_id = a.athlete_id
WHERE rp.predicted_risk_score > 65
  AND rp.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY rp.predicted_risk_score DESC;

COMMENT ON VIEW high_risk_athletes IS 'Shows athletes currently flagged as high risk for injury (last 7 days)';


-- ============================================================================
-- END OF MEDICAL SCHEMA
-- ============================================================================