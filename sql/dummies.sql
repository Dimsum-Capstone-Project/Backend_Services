-- SQLBook: Code
INSERT INTO User (user_id, name, email, password_hash) VALUES
('1', 'John Doe', 'john.doe@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG'),
('2', 'Jane Smith', 'jane.smith@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG'),
('3', 'Alice Johnson', 'alice.johnson@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG'),
('4', 'Bob Brown', 'bob.brown@example.com', '$2b$12$UVsD6CiatJyHwsmxGTTs.O3EmiMWdMec3fHI7LuNEVBihzlpmyibG');

INSERT INTO Profile (profile_id, user_id, job_title, company, bio, profile_picture) VALUES
('1', '1', 'Software Engineer', 'Tech Corp', 'Experienced software engineer.', 'profile_pic_1.jpg'),
('2', '2', 'Product Manager', 'Business Inc', 'Skilled product manager.', 'profile_pic_2.jpg'),
('3', '3', 'Data Scientist', 'Data Corp', 'Expert in data analysis.', 'profile_pic_3.jpg'),
('4', '4', 'UX Designer', 'Design Studio', 'Creative UX designer.', 'profile_pic_4.jpg');

INSERT INTO Contact_Info (contact_id, user_id, contact_type, contact_value, notes) VALUES
('1', '1', 'Phone', '123-456-7890', 'Personal phone number'),
('2', '2', 'Email', 'jane.smith@businessinc.com', 'Work email'),
('3', '3', 'Phone', '987-654-3210', 'Work phone number'),
('4', '4', 'Email', 'bob.brown@designstudio.com', 'Work email'),
('5', '1', 'Email', 'john.doe@techcorp.com', 'Work email'),
('6', '1', 'LinkedIn', 'linkedin.com/in/johndoe', 'LinkedIn profile'),
('7', '2', 'Phone', '555-123-4567', 'Personal phone number'),
('8', '2', 'LinkedIn', 'linkedin.com/in/janesmith', 'LinkedIn profile'),
('9', '3', 'Email', 'alice.johnson@datacorp.com', 'Work email'),
('10', '3', 'LinkedIn', 'linkedin.com/in/alicejohnson', 'LinkedIn profile'),
('11', '4', 'Phone', '444-555-6666', 'Personal phone number'),
('12', '4', 'LinkedIn', 'linkedin.com/in/bobbrown', 'LinkedIn profile');

INSERT INTO Palm_Recognition_Activity (recognition_id, user_id, scanned_user_id, recognition_status, time_scanned) VALUES
('1', '1', '2', TRUE, '2023-01-01 10:00:00'),
('2', '2', '1', FALSE, '2023-01-02 11:00:00'),
('3', '3', '4', TRUE, '2023-01-03 12:00:00'),
('4', '4', '3', FALSE, '2023-01-04 13:00:00'),
('5', '1', '3', TRUE, '2023-01-05 14:00:00'),
('6', '3', '1', FALSE, '2023-01-06 15:00:00'),
('7', '2', '4', TRUE, '2023-01-07 16:00:00'),
('8', '4', '2', FALSE, '2023-01-08 17:00:00'),
('9', '1', '4', TRUE, '2023-01-09 18:00:00'),
('10', '4', '1', FALSE, '2023-01-10 19:00:00'),
('11', '2', '3', TRUE, '2023-01-11 20:00:00'),
('12', '3', '2', FALSE, '2023-01-12 21:00:00');

INSERT INTO Analytics (analytics_id, user_id, total_i_scanned, successful_i_scanned, failed_i_scanned, last_time_i_scanned, total_whos_scanned_me, successful_whos_scanned_me, failed_whos_scanned_me, last_time_whos_scanned_me) VALUES
('1', '1', 5, 3, 2, '2023-01-01 10:00:00', 4, 2, 2, '2023-01-01 10:00:00'),
('2', '2', 3, 2, 1, '2023-01-02 11:00:00', 3, 1, 2, '2023-01-02 11:00:00'),
('3', '3', 7, 5, 2, '2023-01-03 12:00:00', 5, 3, 2, '2023-01-03 12:00:00'),
('4', '4', 2, 1, 1, '2023-01-04 13:00:00', 2, 1, 1, '2023-01-04 13:00:00');

INSERT INTO Password_Reset (reset_id, user_id, reset_token, token_expiration, is_used) VALUES
('1', '1', 'reset_token_1', '2023-01-01 12:00:00', FALSE),
('2', '2', 'reset_token_2', '2023-01-02 12:00:00', TRUE),
('3', '3', 'reset_token_3', '2023-01-03 14:00:00', FALSE),
('4', '4', 'reset_token_4', '2023-01-04 14:00:00', TRUE);

INSERT INTO Token_Blacklist (id, token) VALUES
(1, 'blacklisted_token_1'),
(2, 'blacklisted_token_2'),
(3, 'blacklisted_token_3'),
(4, 'blacklisted_token_4');