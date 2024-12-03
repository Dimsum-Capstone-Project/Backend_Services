CREATE TABLE User (
    user_id CHAR(36) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255)
);

CREATE TABLE Profile (
    profile_id CHAR(36) PRIMARY KEY,
    user_id CHAR(36),
    job_title VARCHAR(255),
    company VARCHAR(255),
    bio TEXT,
    profile_picture VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

CREATE TABLE Contact_Info (
    contact_id CHAR(36) PRIMARY KEY,
    user_id CHAR(36),
    contact_type VARCHAR(255),
    contact_value VARCHAR(255),
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

CREATE TABLE Palm_Recognition_Activity (
    recognition_id CHAR(36) PRIMARY KEY,
    user_id CHAR(36),
    scanned_user_id CHAR(36),
    recognition_status BOOLEAN,
    time_scanned TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

CREATE TABLE Analytics (
    analytics_id CHAR(36) PRIMARY KEY,
    user_id CHAR(36),
    total_scanned INTEGER,
    last_scanned_date TIMESTAMP,
    activity_summary JSON,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

CREATE TABLE Password_Reset (
    reset_id CHAR(36) PRIMARY KEY,
    user_id CHAR(36),
    reset_token VARCHAR(255),
    token_expiration TIMESTAMP,
    is_used BOOLEAN,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);

CREATE TABLE Token_Blacklist (
    id INT PRIMARY KEY AUTO_INCREMENT,
    token VARCHAR(255) UNIQUE,
    blacklisted_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert dummy data into User table
INSERT INTO User (user_id, name, email, password_hash) VALUES
('1', 'John Doe', 'john.doe@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG'),
('2', 'Jane Smith', 'jane.smith@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG');

-- Insert more dummy data into User table
INSERT INTO User (user_id, name, email, password_hash) VALUES
('3', 'Alice Johnson', 'alice.johnson@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG'),
('4', 'Bob Brown', 'bob.brown@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG');

-- Insert dummy data into Profile table
INSERT INTO Profile (profile_id, user_id, job_title, company, bio, profile_picture) VALUES
('1', '1', 'Software Engineer', 'Tech Corp', 'Experienced software engineer.', 'profile_pic_1.jpg'),
('2', '2', 'Product Manager', 'Business Inc', 'Skilled product manager.', 'profile_pic_2.jpg');

-- Insert more dummy data into Profile table
INSERT INTO Profile (profile_id, user_id, job_title, company, bio, profile_picture) VALUES
('3', '3', 'Data Scientist', 'Data Corp', 'Expert in data analysis.', 'profile_pic_3.jpg'),
('4', '4', 'UX Designer', 'Design Studio', 'Creative UX designer.', 'profile_pic_4.jpg');

-- Insert dummy data into Contact_Info table
INSERT INTO Contact_Info (contact_id, user_id, contact_type, contact_value, notes) VALUES
('1', '1', 'Phone', '123-456-7890', 'Personal phone number'),
('2', '2', 'Email', 'jane.smith@businessinc.com', 'Work email');

-- Insert more dummy data into Contact_Info table
INSERT INTO Contact_Info (contact_id, user_id, contact_type, contact_value, notes) VALUES
('3', '3', 'Phone', '987-654-3210', 'Work phone number'),
('4', '4', 'Email', 'bob.brown@designstudio.com', 'Work email');

-- Insert dummy data into Palm_Recognition_Activity table
INSERT INTO Palm_Recognition_Activity (recognition_id, user_id, scanned_user_id, recognition_status, time_scanned) VALUES
('1', '1', '2', TRUE, '2023-01-01 10:00:00'),
('2', '2', '1', FALSE, '2023-01-02 11:00:00');

-- Insert more dummy data into Palm_Recognition_Activity table
INSERT INTO Palm_Recognition_Activity (recognition_id, user_id, scanned_user_id, recognition_status, time_scanned) VALUES
('3', '3', '4', TRUE, '2023-01-03 12:00:00'),
('4', '4', '3', FALSE, '2023-01-04 13:00:00');

-- Insert dummy data into Analytics table
INSERT INTO Analytics (analytics_id, user_id, total_scanned, last_scanned_date, activity_summary) VALUES
('1', '1', 5, '2023-01-01 10:00:00', '{"scans": 5}'),
('2', '2', 3, '2023-01-02 11:00:00', '{"scans": 3}');

-- Insert more dummy data into Analytics table
INSERT INTO Analytics (analytics_id, user_id, total_scanned, last_scanned_date, activity_summary) VALUES
('3', '3', 7, '2023-01-03 12:00:00', '{"scans": 7}'),
('4', '4', 2, '2023-01-04 13:00:00', '{"scans": 2}');

-- Insert dummy data into Password_Reset table
INSERT INTO Password_Reset (reset_id, user_id, reset_token, token_expiration, is_used) VALUES
('1', '1', 'reset_token_1', '2023-01-01 12:00:00', FALSE),
('2', '2', 'reset_token_2', '2023-01-02 12:00:00', TRUE);

-- Insert more dummy data into Password_Reset table
INSERT INTO Password_Reset (reset_id, user_id, reset_token, token_expiration, is_used) VALUES
('3', '3', 'reset_token_3', '2023-01-03 14:00:00', FALSE),
('4', '4', 'reset_token_4', '2023-01-04 14:00:00', TRUE);

-- Insert dummy data into Token_Blacklist table
INSERT INTO Token_Blacklist (id, token) VALUES
(1, 'blacklisted_token_1'),
(2, 'blacklisted_token_2');

-- Insert more dummy data into Token_Blacklist table
INSERT INTO Token_Blacklist (id, token) VALUES
(3, 'blacklisted_token_3'),
(4, 'blacklisted_token_4');