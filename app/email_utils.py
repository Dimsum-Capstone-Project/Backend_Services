
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_reset_email(to_email: str, reset_token: str):
    from_email = "mnyasin26@gmail.com"
    from_password = "vhqi blsd boai duvi"
    subject = "Password Reset Request"
    body = f"Please use the following token to reset your password: {reset_token}"

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")