from twilio.rest import Client
import json
import os
from datetime import datetime
notificationfiles= ['/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-1/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-3/labadmin/Documents/Pybpod/Notifications/notifications.json']
sentnotificationfile = '/home/rozmar/Data/Behavior/Notifications/sent_notifications.json'
experimenters = {'rozsam':'+15716659576','Tina':'+12023694814','NT':'3148562132'}

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
            if notificationnow not in notificationssent and notificationnow['experimenter_name'] in experimenters.keys() :
                print('sending text to ' +notificationnow['experimenter_name'] +  ' at ' + datetime.now().strftime("%Y/%m/%d, %H:%M:%S") + ' by ' + notificationnow['subject_name'])

                experimenter = experimenters[notificationnow['experimenter_name']]
                body = 'Dear '+ notificationnow['experimenter_name'] + ',\n Please take a look at me, I just had ' + notificationnow['reason'] + ' in the ' + notificationnow['setup_name'] + ' rig at '+ notificationnow['datetime'] + '. \nThanks, \n\n'+ notificationnow['subject_name']
                client = Client()
                message = client.messages \
                                .create(
                                     body=body,
                                     from_='whatsapp:+14155238886',
                                     to='whatsapp:'+experimenter
                                 )
                
                print(message.sid)
                
                notificationssent.append(notificationnow)
# =============================================================================
# with open(sentnotificationfile, 'w') as outfile:
#     json.dump(notificationssent, outfile)
# =============================================================================


