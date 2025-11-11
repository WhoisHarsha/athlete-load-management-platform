# Athlete Load Management & Performance Optimization Platform

## DS 5110 Final Project
**Team Members:** Harsha Prakash, Samuel Greeman  
**Instructor:** Dr. Fatema Nafa  
**Northeastern University - Fall 2025**

## Project Overview
Integrated data management system combining sports performance metrics with medical recovery data to predict and prevent athlete injuries.

## Repository Structure
- data/           # Data files and generation scripts
- database/       # SQL schemas and queries
- models/         # ML model code
- notebooks/      # Jupyter notebooks
- docs/           # Documentation
- reports/        # Final reports

## Setup Instructions
1. Install PostgreSQL 14+
2. Install Python 3.9+
3. Install dependencies: pip install -r requirements.txt
4. Run database setup: psql -f database/schemas/create_tables.sql
5. Generate data: python data/scripts/generate_data.py
