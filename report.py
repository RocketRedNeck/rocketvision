import smtplib, ssl

sender_email = "hare.of.caerbannog@gmail.com"
receiver_email = ["5209777127@vtext.com", "kesselmt@gmail.com"]
password = "beau1191!"

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