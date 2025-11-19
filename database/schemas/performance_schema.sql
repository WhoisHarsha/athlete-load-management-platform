DROP TABLE IF EXISTS training_sessions CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS load_calculations CASCADE;

-- ============================================================================
-- TRAINING SESSIONS TABLE
-- ============================================================================
CREATE TABLE training_sessions (
    session_id VARCHAR(15) PRIMARY KEY,
    athlete_id VARCHAR(10) NOT NULL,
    session_date DATE NOT NULL,
    session_type VARCHAR(30) NOT NULL 
        CHECK(session_type IN ('Endurance', 'Speed', 'Strength', 'HIIT', 
                                'Skills', 'Recovery', 'Match/Competition')),
    duration_minutes DECIMAL(6,2) NOT NULL CHECK(duration_minutes > 0),
    start_time TIME,
    end_time TIME,
    intensity_level DECIMAL(4,2) CHECK(intensity_level BETWEEN 1 AND 10),
    rpe DECIMAL(4,2) CHECK(rpe BETWEEN 1 AND 10),
    temperature_celsius DECIMAL(5,2),
    humidity_percent DECIMAL(5,2),
    altitude_meters DECIMAL(7,2),
    surface_type VARCHAR(50),
    coach_notes TEXT,
    athlete_feedback TEXT,
    
    CONSTRAINT fk_training_athlete 
        FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_training_athlete ON training_sessions(athlete_id);
CREATE INDEX idx_training_date ON training_sessions(session_date);
CREATE INDEX idx_training_type ON training_sessions(session_type);

-- ============================================================================
-- PERFORMANCE METRICS TABLE
-- ============================================================================
CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    session_id VARCHAR(15) NOT NULL UNIQUE,
    total_distance_km DECIMAL(6,2) CHECK(total_distance_km >= 0),
    high_speed_distance_km DECIMAL(6,2),
    sprint_distance_km DECIMAL(6,2),
    sprint_count INT CHECK(sprint_count >= 0),
    max_speed_kmh DECIMAL(5,2),
    avg_speed_kmh DECIMAL(5,2),
    acceleration_count INT,
    deceleration_count INT,
    total_player_load DECIMAL(8,2),
    avg_heart_rate INT CHECK(avg_heart_rate > 0),
    max_heart_rate INT CHECK(max_heart_rate > 0),
    min_heart_rate INT,
    hr_zone1_minutes DECIMAL(6,2),
    hr_zone2_minutes DECIMAL(6,2),
    hr_zone3_minutes DECIMAL(6,2),
    hr_zone4_minutes DECIMAL(6,2),
    hr_zone5_minutes DECIMAL(6,2),
    avg_heart_rate_percent DECIMAL(5,2),
    training_impulse DECIMAL(8,2),
    ground_contact_time_ms DECIMAL(6,2),
    vertical_oscillation_cm DECIMAL(5,2),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_perf_session 
        FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_perf_session ON performance_metrics(session_id);

-- ============================================================================
-- LOAD CALCULATIONS TABLE
-- ============================================================================
CREATE TABLE load_calculations (
    load_id SERIAL PRIMARY KEY,
    session_id VARCHAR(15) NOT NULL UNIQUE,
    session_load DECIMAL(10,2) NOT NULL,
    acute_load DECIMAL(10,2),
    chronic_load DECIMAL(10,2),
    acwr DECIMAL(5,2) CHECK(acwr >= 0),
    acwr_category VARCHAR(20) 
        CHECK(acwr_category IN ('Very Low', 'Low', 'Optimal', 'Elevated', 'High Risk')),
    training_monotony DECIMAL(5,2),
    training_strain DECIMAL(10,2),
    weekly_load_change DECIMAL(10,2),
    weekly_load_change_percent DECIMAL(6,2),
    cumulative_7day_load DECIMAL(10,2),
    cumulative_28day_load DECIMAL(10,2),
    is_spike BOOLEAN,
    spike_magnitude DECIMAL(6,2),
    calculation_date DATE NOT NULL,
    days_training_history INT,
    
    CONSTRAINT fk_load_session 
        FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_load_session ON load_calculations(session_id);
CREATE INDEX idx_load_acwr ON load_calculations(acwr);