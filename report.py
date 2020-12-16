import json
import os
import smtplib, ssl
import time

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
    while True:
        try:
            server.login(sender_email, password)
            break
        except Exception as e:
            print(f'[WARNING] {repr(e)}')
            print('Trying SMTP login again')
            time.sleep(1.0)
    server.sendmail(sender_email, receiver_email, message)