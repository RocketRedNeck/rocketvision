import json
import os
import smtplib, ssl

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config_file(file):
    with open(os.path.join(FILE_DIR, file)) as path:
        return json.load(path)



config = load_config_file("hare.json")['hare_options']

sender_email = config['email_source']
receiver_email = config['email_destination']
password = config['password']

message = """\
Subject: Front Door Alert
The Security Rabbit Saw Something."""

port = 465  # For SSL
#password = input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)