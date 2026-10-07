-- 1. Create Clean Fresh Database for LUNG VISION
DROP DATABASE IF EXISTS lungvision_db;
CREATE DATABASE lungvision_db;
USE lungvision_db;

-- 2. Master Users Table (Admin, Doctor, Patient Auth)
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'doctor', 'patient') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Patient Master Directory (Modules 3, 4, 5, 12)
CREATE TABLE patient_records (
    patient_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender ENUM('Male', 'Female', 'Other') NOT NULL,
    phone VARCHAR(20) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    last_visit DATE DEFAULT (CURRENT_DATE),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Radiograph Examination Studies (Modules 6 to 11)
CREATE TABLE radiograph_studies (
    study_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    doctor_name VARCHAR(100) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    champion_model VARCHAR(50) NOT NULL,
    predicted_disease VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 2) NOT NULL,
    risk_score INT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    clinical_recommendation TEXT,
    examination_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patient_records(patient_id) ON DELETE CASCADE
);

-- 5. Admin Live Activity Logs (Module 2)
CREATE TABLE activity_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    message VARCHAR(255) NOT NULL,
    logged_time VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =======================================================
-- INITIAL SEED DATA 
-- =======================================================

-- Sample Registered Patients 
INSERT INTO patient_records (patient_id, name, age, gender, phone, email, last_visit) VALUES
('P001', 'Rahul Verma', 45, 'Male', '9876543210', 'rahul.verma@email.com', '2026-05-10'),
('P002', 'Anita Singh', 38, 'Female', '9123456780', 'anita.s@email.com', '2026-05-09'),
('P003', 'Suresh Kumar', 60, 'Male', '9988776655', 'suresh.k@email.com', '2026-05-08'),
('P004', 'Neha Patel', 29, 'Female', '9876123456', 'neha.p@email.com', '2026-05-07'),
('P005', 'Mohan Das', 52, 'Male', '9090909090', 'mohan.d@email.com', '2026-05-06');

-- Admin Audit Activities 
INSERT INTO activity_logs (message, logged_time) VALUES
('Dr. Gayathri uploaded a report', '10:30 AM'),
('New patient registered', '09:45 AM'),
('AI analysis completed', '09:15 AM'),
('Report generated', '08:50 AM');
SET SQL_SAFE_UPDATES = 0;
DELETE FROM users WHERE role = 'admin';
SET SQL_SAFE_UPDATES = 1;
USE lungvision_db;


USE lungvision_db;
DESCRIBE users;

USE lungvision_db;


SET SQL_SAFE_UPDATES = 0;
DELETE FROM users WHERE role = 'admin';
SET SQL_SAFE_UPDATES = 1;


INSERT INTO users (name, email, password_hash, role) 
VALUES ('System Administrator', 'admin@gmail.com', '123456', 'admin');


SELECT user_id, name, email, password_hash, role FROM users;

SELECT COUNT(*) FROM pulmoverse_db.radiograph_studies;
USE lungvision_db;

ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';

SELECT user_id, name, email, role, status FROM users;
USE lungvision_db;

SELECT * FROM users;
SELECT * FROM patients;
SELECT * FROM doctors;
SELECT * FROM users WHERE role = 'patient';
SELECT * FROM users WHERE role = 'doctor';
SELECT * FROM users WHERE role = 'admin';
SELECT user_id, name, email, role, status FROM users;
SELECT * FROM patient_records;
SHOW TABLES;
SET FOREIGN_KEY_CHECKS = 0;

-- 1. Patient records-la mathunga:
UPDATE patient_records SET patient_id = 'P010' WHERE patient_id = 'PT-8';

-- 2. Radiograph studies-layum athe ID-ai mathunga:
UPDATE radiograph_studies SET patient_id = 'P010' WHERE patient_id = 'PT-8';

SET FOREIGN_KEY_CHECKS = 1;
SELECT * FROM radiograph_studies;
SELECT 
    r.study_id,
    p.patient_id,
    p.name AS patient_name,
    p.age,
    p.gender,
    r.prediction,
    r.confidence,
    r.created_at
FROM radiograph_studies r
JOIN patient_records p ON r.patient_id = p.patient_id;
SELECT * FROM patient_records;

USE lungvision_db;

SELECT * 
FROM radiograph_studies r 
JOIN patient_records p ON r.patient_id = p.patient_id;
DESCRIBE radiograph_studies;
