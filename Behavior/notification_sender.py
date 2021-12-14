import json
import os
import smtplib
from datetime import datetime
notificationfiles= ['/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-1/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-3/labadmin/Documents/Pybpod/Notifications/notifications.json']
sentnotificationfile = '/home/rozmar/Data/Behavior/Notifications/sent_notifications.json'

gmail_user = 'mouse.watcher007@gmail.com'
gmail_password = 'passwordformousewatcher007'
experimenters = {'rozsam':'qtyush@gmail.com','Tina':'tkpluntke@gmail.com','NT':'tienn@janelia.hhmi.org'}

if os.path.exists(sentnotificationfile):
    with open(sentnotificationfile) as json_file:
        notificationssent = json.load(json_file)
else:
    notificationssent  = list()
    

for notificationfile in notificationfiles:
    if os.path.exists(notificationfile):
        with open(notificationfile) as json_file:
            notificationsnow = json.load(json_file)
            
        for notificationnow in notificationsnow:
            if notificationnow not in notificationssent:
                print('sending email to ' +notificationnow['experimenter_name'] +  ' at ' + datetime.now().strftime("%Y/%m/%d, %H:%M:%S")) + ' by ' + notificationnow['subject_name']
                sent_from = gmail_user
                to = [experimenters[notificationnow['experimenter_name']]]
                subject =  notificationnow['subject_name'] + ' needs help'
                body = 'Dear '+ notificationnow['experimenter_name'] + ',\n Please take a look at me, I just had ' + notificationnow['reason'] + ' in the ' + notificationnow['setup_name'] + ' rig at '+ notificationnow['datetime'] + '. \nThanks, \n\n'+ notificationnow['subject_name']
                
                email_text = 'From: {}\nTo: {}\nSubject: {}\n\n{}'.format(sent_from, ", ".join(to), subject, body)
    
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.ehlo()
                server.login(gmail_user, gmail_password)
                server.sendmail(sent_from, to, email_text)
                server.close()
                notificationssent.append(notificationnow)
with open(sentnotificationfile, 'w') as outfile:
    json.dump(notificationssent, outfile)
