#%%

import slack

import json
import os
from datetime import datetime
import numpy as np
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
#SLACK_BOT_TOKEN = 'xoxb-60282475316-834537704051-4ESSsQBjvsS0ZHiuQBLaqfBC'
#SLACK_BOT_TOKEN = 'a'
notificationfiles= ['/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-1/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Notifications/notifications.json',
                    '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-3/labadmin/Documents/Pybpod/Notifications/notifications.json']
sentnotificationfile = '/home/rozmar/Data/Behavior/Notifications/sent_notifications-'+datetime.now().strftime('%Y-%m-%d')+'.json'
experimenters = {'rozsam':'DQH6LT8SE','Tina':'DQSMAK5CZ','NT':' '}
fullnames = {'rozsam':'<@UJWK9ETM1|cal>','Tina':'<@U1TMFPZDL|cal>','NT':'<@UCELR6CLW|cal>'}
#%%
if os.path.exists(sentnotificationfile):
    with open(sentnotificationfile) as json_file:
        notificationssent = json.load(json_file)
else:
    notificationssent  = list()
    
#%
for notificationfile in notificationfiles:
    if os.path.exists(notificationfile):
        with open(notificationfile) as json_file:
            notificationsnow = json.load(json_file)
            
        for notificationnow in notificationsnow:
            if notificationnow not in notificationssent and notificationnow['experimenter_name'] in experimenters.keys() :
                
                #print(notificationnow)
                experimenter = experimenters[notificationnow['experimenter_name']]
                body = fullnames[notificationnow['experimenter_name']]+'!!' + ',\n' + notificationnow['subject_name'] + ' had ' + notificationnow['reason'] + ' in the ' + notificationnow['setup_name'] + ' rig at '+ notificationnow['datetime'] + '.'
                
                
                
                timediff = datetime.now() - datetime.strptime(notificationnow['datetime'],'%Y/%m/%d, %H:%M:%S')
                
                if np.abs(timediff.seconds)<3600 and np.abs(timediff.days)<1: # only recent notifications will be sent
                    print('sending slack message to ' +notificationnow['experimenter_name'] +  ' at ' + datetime.now().strftime("%Y/%m/%d, %H:%M:%S") + ' by ' + notificationnow['subject_name'])
                    client = slack.WebClient(token=SLACK_BOT_TOKEN)
                    try:
                        response = client.chat_postMessage(
                            channel=experimenter,
                            text=body)                
                    except:
                        print('error sending slack message')
                        pass
                    notificationssent.append(notificationnow)
with open(sentnotificationfile, 'w') as outfile: # the email won't be sent
    json.dump(notificationssent, outfile)


