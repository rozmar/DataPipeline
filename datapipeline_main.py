import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from pybpodgui_api.models.project import Project
import numpy as np
import Behavior.behavior_rozmar as behavior_rozmar
#%% connect to server
import datajoint as dj
dj.conn()
from pipeline import lab, experiment

#%% add users
print('adding experimenters')
experimenterdata = [
        {
        'username' : 'rozsam',
        'fullname' : 'Marton Rozsa'
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
        'rig' : 'Behavior-Tower-2',
        'room' : '2w.339',
        'rig_description' : 'training rig'
        }        
        ]
for rignow in rigdata:
    try:
        lab.Rig().insert1(rignow)
    except dj.DuplicateError:
        print('duplicate. rig: ',rignow['rig'], ' already exists')
#%% populate subjects, surgeries and water restrictions
print('adding surgeries and stuff')
df_surgery = online_notebook.fetch_animal_metadata()
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
    while 'surgery date ('+str(surgeryidx)+')' in item[1].keys() and item[1]['surgery date ('+str(surgeryidx)+')']:
        start_time = datetime.strptime(item[1]['surgery date ('+str(surgeryidx)+')']+' '+item[1]['surgery time ('+str(surgeryidx)+')'],'%Y-%m-%d %H:%M')
        end_time = start_time + timedelta(minutes = int(item[1]['surgery length (min) ('+str(surgeryidx)+')']))
        surgerydata = {
                'surgery_id': surgeryidx,
                'subject_id':item[1]['animal#'],
                'username': item[1]['experimenter'],
                'start_time': start_time,
                'end_time': end_time,
                'surgery_description': item[1]['surgery type ('+str(surgeryidx)+')'] + ':-: comments: ' + item[1]['surgery comments ('+str(surgeryidx)+')'],
                }
        surgeryidx += 1
        try:
            lab.Surgery().insert1(surgerydata)
        except dj.DuplicateError:
            print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ', start_time)
    
    if item[1]['ID']:
        df_wr = online_notebook.fetch_water_restriction_metadata(item[1]['ID'])
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
                  
#%% populate water restriction and training days
#%% load pybpod data
print('adding behavior experiments')
directories = dict()
directories = {'behavior_project_dirs' : ['/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects/Foraging',
                                          '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects/Foraging_again']
    }
projects = list()
for projectdir in directories['behavior_project_dirs']:
    projects.append(Project())
    projects[-1].load(projectdir)


IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
for subject_now in IDs.keys(): # iterating over subjects
    print('subject: ',subject_now)
    df_wr = online_notebook.fetch_water_restriction_metadata(subject_now)
    
    for df_wr_row in df_wr.iterrows():
        if df_wr_row[1]['Time'] and df_wr_row[1]['Time-end']: # we use it when both start and end times are filled in
            date_now = df_wr_row[1].Date.replace('-','')
            print('date: ',date_now)
            #%% ingest metadata
            
            #%%
            sessions_now = list()
            session_start_times_now = list()
            for proj in projects: #
                exps = proj.experiments
                for exp in exps:
                    stps = exp.setups
                    for stp in stps:
                        sessions = stp.sessions
                        for session in stp.sessions:
                            if session.subjects and session.subjects[0].find(subject_now) >-1 and session.name.startswith(date_now):
                                sessions_now.append(session)
                                session_start_times_now.append(session.started)
                                
            order = np.argsort(session_start_times_now)
            for session_idx in order:
                session = sessions_now[session_idx]
                csvfilename = (Path(session.path) / (Path(session.path).name + '.csv'))
                df_behavior_session = behavior_rozmar.load_and_parse_a_csv_file(csvfilename)










# =============================================================================
# #%% Drop everything
# schemastodrop = [lab,experiment]
# for shemanow in schemastodrop:
#     for attr, value in shemanow.__dict__.items():
#         if attr[0].isupper():
#             try:
#                 value.drop()
#             except:
#                 print('already dropped')
# =============================================================================
# =============================================================================
# #%% Drop everythink      
# schema = dj.schema('rozmar_foraging-experiment')
# schema.drop(force=True)
# schema = dj.schema('rozmar_foraging-lab')
# schema.drop(force=True)    
#                 
# =============================================================================






    