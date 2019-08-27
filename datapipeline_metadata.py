import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
import json
import time as timer
#% connect to server
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment

def populatemetadata():
    #%% save metadata from google drive if necessairy
    lastmodify = online_notebook.fetch_lastmodify_time_animal_metadata()
    with open(dj.config['locations.metadata']+'last_modify_time.json') as timedata:
        lastmodify_prev = json.loads(timedata.read())
    if lastmodify != lastmodify_prev:
        print('updating surgery and WR metadata from google drive')
        dj.config['locations.metadata']
        df_surgery = online_notebook.fetch_animal_metadata()
        df_surgery.to_csv(dj.config['locations.metadata']+'Surgery.csv')
        IDs = df_surgery['ID'].tolist()
        for ID in IDs:
            df_wr = online_notebook.fetch_water_restriction_metadata(ID)
            df_wr.to_csv(dj.config['locations.metadata']+ID+'.csv') 
        with open(dj.config['locations.metadata']+'last_modify_time.json', "w") as write_file:
            json.dump(lastmodify, write_file)
        print('surgery and WR metadata updated')
    #%% add users
    print('adding experimenters')
    experimenterdata = [
            {
            'username' : 'rozsam',
            'fullname' : 'Marton Rozsa'
            },
            {
            'username' : 'Tina',
            'fullname' : 'Tina Pluntke'
            }
            ]
    for experimenternow in experimenterdata:
        try:
            lab.Person().insert1(experimenternow)
        except dj.DuplicateError:
            print('duplicate. experimenter: ',experimenternow['username'], ' already exists')
    
    #%% add rigs
    print('adding rigs')
    rigdata = [
            {
            'rig' : 'Training-Tower-2',
            'room' : '2w.339',
            'rig_description' : 'training rig'
            },        
            {
            'rig' : 'Training-Tower-3',
            'room' : '2w.339',
            'rig_description' : 'training rig'
            },
            {
            'rig' : 'Voltage-Imaging-1p',
            'room' : '2w.333',
            'rig_description' : '1p voltage imaging, behavior, whole cell patch clamp'
            },        
            ]
    for rignow in rigdata:
        try:
            lab.Rig().insert1(rignow)
        except dj.DuplicateError:
            print('duplicate. rig: ',rignow['rig'], ' already exists')
    #%% populate subjects, surgeries and water restrictions
    print('adding surgeries and stuff')
    df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
    for item in df_surgery.iterrows():
        subjectdata = {
                'subject_id': item[1]['animal#'],
                'cage_number': item[1]['cage#'],
                'date_of_birth': item[1]['DOB'],
                'sex': item[1]['sex'],
                'username': item[1]['experimenter'],
                }
        try:
            lab.Subject().insert1(subjectdata)
        except dj.DuplicateError:
            print('duplicate. animal :',item[1]['animal#'], ' already exists')
        
        
        surgeryidx = 1
        while 'surgery date ('+str(surgeryidx)+')' in item[1].keys() and item[1]['surgery date ('+str(surgeryidx)+')'] and type(item[1]['surgery date ('+str(surgeryidx)+')']) == str:
            start_time = datetime.strptime(item[1]['surgery date ('+str(surgeryidx)+')']+' '+item[1]['surgery time ('+str(surgeryidx)+')'],'%Y-%m-%d %H:%M')
            end_time = start_time + timedelta(minutes = int(item[1]['surgery length (min) ('+str(surgeryidx)+')']))
            surgerydata = {
                    'surgery_id': surgeryidx,
                    'subject_id':item[1]['animal#'],
                    'username': item[1]['experimenter'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'surgery_description': item[1]['surgery type ('+str(surgeryidx)+')'] + ':-: comments: ' + str(item[1]['surgery comments ('+str(surgeryidx)+')']),
                    }
            surgeryidx += 1
            try:
                lab.Surgery().insert1(surgerydata)
            except dj.DuplicateError:
                print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ', start_time)
        
        if item[1]['ID']:
            #df_wr = online_notebook.fetch_water_restriction_metadata(item[1]['ID'])
            df_wr = pd.read_csv(dj.config['locations.metadata']+item[1]['ID']+'.csv')
            if type(df_wr) == pd.DataFrame:
                wrdata = {
                        'subject_id':item[1]['animal#'],
                        'water_restriction_number': item[1]['ID'],
                        'cage_number': item[1]['cage#'],
                        'wr_start_date': df_wr['Date'][0],
                        'wr_start_weight': df_wr['Weight'][0],
                        }
            try:
                lab.WaterRestriction().insert1(wrdata)
            except dj.DuplicateError:
                print('duplicate. water restriction :',item[1]['animal#'], ' already exists')
                      