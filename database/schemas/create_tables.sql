-- Athlete Load Management Platform
-- Database Schema Creation Script

-- Athletes table (shared)
CREATE TABLE IF NOT EXISTS athletes (
    athlete_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER,
    position VARCHAR(50),
    team VARCHAR(50)
);

-- Medical tables (Harsha)
CREATE TABLE IF NOT EXISTS injuries (
    injury_id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(athlete_id),
    injury_date DATE,
    injury_type VARCHAR(100),
    severity INTEGER,
    muscle_group VARCHAR(50),
    recovery_days INTEGER
);

CREATE TABLE IF NOT EXISTS recovery_metrics (
    metric_id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(athlete_id),
    date_recorded DATE,
    sleep_hours DECIMAL(3,1),
    soreness_level INTEGER,
    fatigue_level INTEGER
);

-- Performance tables (Samuel)
CREATE TABLE IF NOT EXISTS training_sessions (
    session_id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(athlete_id),
    session_date DATE,
    duration_minutes INTEGER,
    distance_km DECIMAL(5,2),
    intensity_level INTEGER
);

CREATE TABLE IF NOT EXISTS load_calculations (
    calc_id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(athlete_id),
    week_number INTEGER,
    acwr DECIMAL(3,2),
    monotony DECIMAL(4,2),
    strain DECIMAL(6,2)
);
