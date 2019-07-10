import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from pybpodgui_api.models.project import Project
import numpy as np
import Behavior.behavior_rozmar as behavior_rozmar
#%% connect to server
import datajoint as dj
dj.config['database.host'] = 'mesoscale-activity.datajoint.io'
dj.config['database.user'] = 'rozmar'
dj.config['database.password'] = 'new-account-for-new-data'

dj.conn()
from pipeline import lab
subject = lab.Subject()
surgery = lab.Surgery()
wr = lab.WaterRestriction()
#%% populate subjects, surgeries and water restrictions
df = online_notebook.fetch_animal_metadata()
for item in df.iterrows():
    subjectdata = {
            'subject_id': item[1]['animal#'],
            'cage_number': item[1]['cage#'],
            'date_of_birth': item[1]['DOB'],
            'sex': item[1]['sex'],
            'username': item[1]['experimenter'],
            }
    try:
        subject.insert1(subjectdata)
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
            surgery.insert1(surgerydata)
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
            wr.insert1(wrdata)
        except dj.DuplicateError:
            print('duplicate. water restriction :',item[1]['animal#'], ' already exists')
                  
#%% populate water restriction and training days
directories = dict()
directories = {'behavior_project_dirs' : ['/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects/Foraging',
                                          '/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects/Foraging_again']
    }
projects = list()
for projectdir in directories['behavior_project_dirs']:
    projects.append(Project())
    projects[-1].load(projectdir)
#proj.load(directories['behavior_project_dirs'][1])
#%% load pybpod data
subject_now = 'FOR04'
date_now = '20190703'
sessions_now = list()
session_start_times_now = list()
for proj in projects:
    # =============================================================================
    # subjects = proj.subjects
    # for subject in subjects: 
    #     if subject.name == subject_now:
    # =============================================================================
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
    df = behavior_rozmar.load_and_parse_a_csv_file(csvfilename)
    print(sessions_now[session_idx].name)
    break
# =============================================================================
#                     
#                     if csvfilename.is_file():
#                         
#                         #
#                         print('load')   
#                     else:
#                         print(session.path)
# =============================================================================
    
    