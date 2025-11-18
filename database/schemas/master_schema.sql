-- ============================================================================
-- ATHLETE MASTER TABLE SCHEMA
-- ============================================================================
-- Purpose: Central table containing core athlete information and demographics
-- Author: Harsha Prakash (Health Data Science)
-- Date: November 2025
-- Project: Athlete Load Management & Performance Optimization Platform
-- ============================================================================

-- Drop existing table if exists (for development/testing)
DROP TABLE IF EXISTS athletes CASCADE;

-- Create athletes master table
CREATE TABLE athletes (
    -- Primary identifier
    athlete_id VARCHAR(10) PRIMARY KEY,
    
    -- Personal Information
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    
    -- Athletic Profile
    position VARCHAR(30) NOT NULL, -- e.g., Forward, Midfielder, Defender, Goalkeeper
    dominant_side VARCHAR(10) CHECK (dominant_side IN ('Left', 'Right', 'Both')),
    years_experience INT CHECK (years_experience >= 0),
    height_cm DECIMAL(5,2) CHECK (height_cm > 0 AND height_cm < 250),
    weight_kg DECIMAL(5,2) CHECK (weight_kg > 0 AND weight_kg < 200),
    
    -- Team Information
    team_name VARCHAR(50),
    jersey_number INT CHECK (jersey_number > 0 AND jersey_number <= 99),
    contract_start_date DATE,
    contract_end_date DATE,
    
    -- Status Tracking
    current_status VARCHAR(20) DEFAULT 'Active' 
        CHECK (current_status IN ('Active', 'Injured', 'Recovering', 'Retired', 'Suspended')),
    fitness_level VARCHAR(20) DEFAULT 'Good'
        CHECK (fitness_level IN ('Excellent', 'Good', 'Fair', 'Poor')),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_contract_dates CHECK (contract_end_date >= contract_start_date)
);

-- Create indexes for frequently queried columns
CREATE INDEX idx_athletes_position ON athletes(position);
CREATE INDEX idx_athletes_status ON athletes(current_status);
CREATE INDEX idx_athletes_team ON athletes(team_name);
CREATE INDEX idx_athletes_dob ON athletes(date_of_birth);

-- Add comments to document the table
COMMENT ON TABLE athletes IS 'Master table containing core athlete demographics and profile information';
COMMENT ON COLUMN athletes.athlete_id IS 'Unique identifier for each athlete (format: ATH_XXX)';
COMMENT ON COLUMN athletes.date_of_birth IS 'Date of birth used to calculate age when needed';
COMMENT ON COLUMN athletes.position IS 'Primary playing position on the field';
COMMENT ON COLUMN athletes.years_experience IS 'Total years of professional/competitive experience';
COMMENT ON COLUMN athletes.current_status IS 'Current availability status for training/competition';
COMMENT ON COLUMN athletes.fitness_level IS 'Overall fitness assessment by medical staff';

-- Create trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_athlete_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_athlete_timestamp
BEFORE UPDATE ON athletes
FOR EACH ROW
EXECUTE FUNCTION update_athlete_timestamp();

-- ============================================================================
-- SAMPLE DATA INSERTION (Optional - for testing)
-- ============================================================================
-- Uncomment below to insert sample athletes

/*
INSERT INTO athletes (
    athlete_id, first_name, last_name, date_of_birth, gender, 
    position, dominant_side, years_experience, height_cm, weight_kg,
    team_name, jersey_number, contract_start_date, contract_end_date
) VALUES
    ('ATH_001', 'John', 'Smith', '1995-03-15', 'Male', 'Forward', 'Right', 8, 182.5, 78.5, 'Team Alpha', 10, '2023-01-01', '2026-12-31'),
    ('ATH_002', 'Sarah', 'Johnson', '1998-07-22', 'Female', 'Midfielder', 'Right', 5, 168.0, 62.0, 'Team Alpha', 7, '2023-06-01', '2025-05-31'),
    ('ATH_003', 'Michael', 'Brown', '1993-11-08', 'Male', 'Defender', 'Left', 10, 185.0, 82.0, 'Team Beta', 4, '2022-01-01', '2026-12-31');
*/

-- ============================================================================
-- END OF MASTER SCHEMA
-- ============================================================================