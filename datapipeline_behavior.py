
from datetime import datetime, timedelta, time
import pandas as pd
from pathlib import Path
from pybpodgui_api.models.project import Project
import numpy as np
import Behavior.behavior_rozmar as behavior_rozmar
import json
import time as timer
import decimal
#% connect to server
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment
from pipeline import behavioranal
  
def populatemytables():
    arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
    behavioranal.TrialReactionTime().populate(**arguments)
    behavioranal.SessionReactionTimeHistogram().populate(**arguments)
    behavioranal.SessionLickRhythmHistogram().populate(**arguments)  
    behavioranal.SessionTrainingType().populate(**arguments)  
    behavioranal.SessionRewardRatio().populate(**arguments)  
    behavioranal.BlockRewardRatio().populate(**arguments)  
    behavioranal.SessionBlockSwitchChoices().populate(**arguments)  
    behavioranal.SessionFittedChoiceCoefficients().populate(**arguments)



def populatebehavior():
    df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
    #%% load pybpod data
    print('adding behavior experiments')
    directories = dict()
    directories = {'behavior_project_dirs' : ['/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging_again',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging_homecage',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-3/Foraging_homecage']
        }
    projects = list()
    for projectdir in directories['behavior_project_dirs']:
        projects.append(Project())
        projects[-1].load(projectdir)
    
    
    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
    for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects
        print('subject: ',subject_now)
        delete_last_session_before_upload = True
        #df_wr = online_notebook.fetch_water_restriction_metadata(subject_now)
        df_wr = pd.read_csv(dj.config['locations.metadata']+subject_now+'.csv')
        for df_wr_row in df_wr.iterrows():
            if df_wr_row[1]['Time'] and df_wr_row[1]['Time-end'] and df_wr_row[1]['Training type'] != 'restriction' and df_wr_row[1]['Training type'] != 'handling': # we use it when both start and end times are filled in, restriction and handling is skipped
                date_now = df_wr_row[1].Date.replace('-','')
                print('date: ',date_now)
    
                sessions_now = list()
                session_start_times_now = list()
                experimentnames_now = list()
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
                                    experimentnames_now.append(exp.name)
                order = np.argsort(session_start_times_now)
                for session_idx in order:
                    session = sessions_now[session_idx]
                    session_date = datetime(int(date_now[0:4]),int(date_now[4:6]),int(date_now[6:8]))
                    experiment_name = experimentnames_now[session_idx]
                    csvfilename = (Path(session.path) / (Path(session.path).name + '.csv'))
                    if len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date > "'+str(session_date)+'"') != 0:
                        print('session already imported, skipping: ' + str(session_date))
                    else: # reuploading only the LAST session that is present on the server
                        if delete_last_session_before_upload == True and df_surgery['status'][df_surgery['ID']==subject_now].values[0] != 'sacrificed': # the last session is deleted in the animal is still in training..
                            print(df_surgery['status'][df_surgery['ID']==subject_now].values[0])
                            if len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"') != 0:
                                print('dropping last session')
                                session_todel =experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"'
                                dj.config['safemode'] = False
                                session_todel.delete()
                                dj.config['safemode'] = True
                            delete_last_session_before_upload = False
                            
                        df_behavior_session = behavior_rozmar.load_and_parse_a_csv_file(csvfilename)
                        if 'var:WaterPort_L_ch_in' in df_behavior_session.keys():# if the variables are not saved, they are inherited from the previous session
                            channel_L = df_behavior_session['var:WaterPort_L_ch_in'][0]
                            channel_R = df_behavior_session['var:WaterPort_R_ch_in'][0]
                        trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
                        if len(trial_start_idxs)>0: # is there a trial start
                            session_time = df_behavior_session['PC-TIME'][trial_start_idxs[0]]
                            if session.setup_name.lower() in ['day1','tower-2','day2-7','day_1','real foraging']:
                                setupname = 'Training-Tower-2'
                            elif session.setup_name.lower() in ['tower-3']:
                                setupname = 'Training-Tower-3'
                            else:
                                setupname = 'unhandled'
                                print('setup name not handled:'+session.setup_name)                                
                            #% ingest session metadata                
                            sessiondata = {
                                    'subject_id': subject_id_now,#(lab.WaterRestriction() & 'water_restriction_number = "'+df_behavior_session['subject'][0]+'"').fetch()[0]['subject_id'],
                                    'session' : np.nan,
                                    'session_date' : session_date,
                                    'session_time' : session_time.strftime('%H:%M:%S'),
                                    'username' : df_behavior_session['experimenter'][0],
                                    'rig': setupname
                                    }
                            if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"') == 0:
                                if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"') == 0:
                                    sessiondata['session'] = 1
                                else:
                                    sessiondata['session'] = len((experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"').fetch()['session']) + 1
                                experiment.Session().insert1(sessiondata)
                                # ingest session comments
                                sessioncommentdata = {
                                    'subject_id': subject_id_now ,
                                    'session': sessiondata['session'],
                                    'session_comment': str(df_wr_row[1]['Notes'])
                                    }
                                experiment.SessionComment().insert1(sessioncommentdata)
                                
                                sessiondetailsdata = {
                                        'subject_id': subject_id_now ,
                                        'session': sessiondata['session'],
                                        'session_weight': df_wr_row[1]['Weight'],
                                        'session_water_earned' : df_wr_row[1]['Water during training'],
                                        'session_water_extra' : df_wr_row[1]['Extra water']
                                        }
                                experiment.SessionDetails().insert1(sessiondetailsdata)
                            print(date_now + ' - ' + session.task_name)
                            session_now = (experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"').fetch()
                            session_start_time = datetime.combine(session_now['session_date'][0],datetime.min.time()) +session_now['session_time'][0]
                            #% extracting trial data
                            trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
                            trial_end_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'END-TRIAL')].index
                            for trial_start_idx,trial_end_idx in zip(trial_start_idxs,trial_end_idxs) :
                                df_behavior_trial = df_behavior_session[trial_start_idx:trial_end_idx+1]
                                #Trials without GoCue  are skipped
                                if len(df_behavior_trial['PC-TIME'][(df_behavior_trial['MSG'] == 'GoCue') & (df_behavior_trial['TYPE'] == 'TRANSITION')]) > 0:
                
                                    trial_start_time = df_behavior_session['PC-TIME'][trial_start_idx].to_pydatetime() - session_start_time
                                    trial_stop_time = df_behavior_session['PC-TIME'][trial_end_idx].to_pydatetime() - session_start_time
                                    if len(experiment.SessionTrial() & 'session =' + str(session_now['session'][0]) & 'trial_start_time ="' + str(trial_start_time) + '"') == 0:# importing if this trial is not already imported
                                        trialnum = len(experiment.SessionTrial() & 'session =' + str(session_now['session'][0]) & 'subject_id = "' + str(subject_id_now) + '"') + 1
                                        unique_trialnum = len(experiment.SessionTrial()) + 1 
                                        sessiontrialdata={
                                                'subject_id':subject_id_now,
                                                'session':session_now['session'][0],
                                                'trial': trialnum,
                                                'trial_uid': unique_trialnum, # unique across sessions/animals
                                                'trial_start_time': trial_start_time.total_seconds(), # (s) relative to session beginning 
                                                'trial_stop_time': trial_stop_time.total_seconds()# (s) relative to session beginning 
                                                }
                                        experiment.SessionTrial().insert1(sessiontrialdata, allow_direct_insert=True)
                                        
                                        #%%
                                        if 'Block_number' in df_behavior_session.columns and not np.isnan(df_behavior_trial['Block_number'].to_list()[0]):
                                            blocknum_local = int(df_behavior_trial['Block_number'].to_list()[0])-1
                                            p_reward_left = decimal.Decimal(df_behavior_trial['var:reward_probabilities_L'].to_list()[0][blocknum_local]).quantize(decimal.Decimal('.001'))
                                            p_reward_right = decimal.Decimal(df_behavior_trial['var:reward_probabilities_R'].to_list()[0][blocknum_local]).quantize(decimal.Decimal('.001'))
                                            if len(experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0])) == 0:
                                                p_reward_right_previous = -1
                                                p_reward_left_previous = -1
                                            else:
                                                #%%
                                                prevblock =  (experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0])).fetch('block').max()
                                                probs = (experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0]) & 'block = ' + str(prevblock)).fetch('p_reward_left','p_reward_right')
                                                p_reward_right_previous  = probs[1][0]
                                                p_reward_left_previous  = probs[0][0]
                                            
                                            if p_reward_left != p_reward_left_previous or p_reward_right !=p_reward_right_previous:
                                                #%%  
                                                if len(experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0])) == 0:
                                                    blocknum = 1
                                                else:
                                                    blocknum = len(experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0])) + 1
                                                if blocknum>200:
                                                    print('waiting.. there are way too many blocks')
                                                    timer.sleep(1000)
                                                unique_blocknum = len(experiment.SessionBlock()) + 1
                                                block_start_time = trial_start_time.total_seconds()
                                                p_reward_left = p_reward_left
                                                p_reward_right = p_reward_right
                                                sessionblockdata={
                                                    'subject_id':subject_id_now,
                                                    'session':session_now['session'][0],
                                                    'block': blocknum,
                                                    'block_uid': unique_blocknum, # unique across sessions/animals
                                                    'block_start_time': block_start_time, # (s) relative to session beginning 
                                                    'p_reward_left' : p_reward_left,
                                                    'p_reward_right' : p_reward_right
                                                    }
                                                experiment.SessionBlock().insert1(sessionblockdata, allow_direct_insert=True)
                                                print('new block added: ' + str (block_start_time))
                                            blocknum_now = (experiment.SessionBlock() & 'subject_id = "'+str(subject_id_now)+'"' & 'session = ' + str(session_now['session'][0])).fetch('block').max()
                                        else:
                                            blocknum_now = None
                                        #%%
                                        #% extracting task protocol
                                        if experiment_name in ['Bari_Cohen', 'Foraging - Bari-Cohen']:
                                            task = 'foraging'
                                            if 'var:early_lick_punishment' in df_behavior_session.keys():# inherits task protocol if variables are not available
                                                if df_behavior_session['var:early_lick_punishment'][0] and df_behavior_session['var:motor_retract_waterport'][0]:
                                                    task_protocol = 10
                                                elif df_behavior_session['var:motor_retract_waterport'][0]:
                                                    task_protocol = 14
                                                elif df_behavior_session['var:early_lick_punishment'][0]:
                                                    task_protocol = 13
                                                else:
                                                    task_protocol = 12
                                        elif experiment_name in ['Delayed_foraging','Delayed foraging','Foraging_homecage']:
                                            task = 'del foraging'
                                            if 'var:early_lick_punishment' in df_behavior_session.keys():# inherits task protocol if variables are not available
                                                if df_behavior_session['var:early_lick_punishment'][0] and df_behavior_session['var:motor_retract_waterport'][0]:
                                                    task_protocol = 11
                                                elif df_behavior_session['var:motor_retract_waterport'][0]:
                                                    task_protocol = 17
                                                elif df_behavior_session['var:early_lick_punishment'][0]:
                                                    task_protocol = 16
                                                else:
                                                    task_protocol = 15
                                        else:
                                            task = np.nan
                                            task_protocol = 'nan' 
                                            print('task name not handled:'+experiment_name)
                                        
                                        if any((df_behavior_trial['MSG'] == 'Choice_L') & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                                            trial_choice = 'left'
                                        elif any((df_behavior_trial['MSG'] == 'Choice_R') & (df_behavior_trial['TYPE'] == 'TRANSITION')): 
                                            trial_choice = 'right'
                                        else:
                                            trial_choice = 'none'
                                        #%
                                        time_TrialStart = df_behavior_session['PC-TIME'][trial_start_idx]
                                        time_GoCue = df_behavior_trial['PC-TIME'][(df_behavior_trial['MSG'] == 'GoCue') & (df_behavior_trial['TYPE'] == 'TRANSITION')]
                                        time_lick_left = df_behavior_trial['PC-TIME'][(df_behavior_trial['+INFO'] == channel_L)]
                                        time_lick_right = df_behavior_trial['PC-TIME'][(df_behavior_trial['+INFO'] == channel_R)]
                                        if any(time_lick_left.to_numpy() - time_GoCue.to_numpy() <np.timedelta64(0)) or any(time_lick_right.to_numpy() - time_GoCue.to_numpy() <np.timedelta64(0)):
                                            early_lick = 'early'
                                        else:
                                            early_lick = 'no early'
                                        #% outcome
                                        if any((df_behavior_trial['MSG'] == 'Reward_R') & (df_behavior_trial['TYPE'] == 'TRANSITION')) or any((df_behavior_trial['MSG'] == 'Reward_L') & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                                            outcome = 'hit'
                                        elif trial_choice == 'none':
                                            outcome = 'ignore'
                                        else:
                                            outcome = 'miss'
                                        #%
                                        behaviortrialdata = {
                                                'subject_id': subject_id_now,
                                                'session': session_now['session'][0],
                                                'trial': trialnum,
                                                'task': task,
                                                'task_protocol': task_protocol,
                                                #'trial_instruct':,
                                                'trial_choice': trial_choice,
                                                'early_lick':early_lick,
                                                'outcome': outcome
                                                }
                                        if blocknum_now:
                                            behaviortrialdata['block'] = blocknum_now
                                        experiment.BehaviorTrial().insert1(behaviortrialdata, allow_direct_insert=True)
                                        ##%
                                        trialnotedata = None
                                        if any((df_behavior_trial['MSG'] == 'Auto_Water_L') & (df_behavior_trial['TYPE'] == 'TRANSITION')) and any((df_behavior_trial['MSG'] == 'Auto_Water_R') & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                                            trialnotedata = {
                                                  'subject_id': subject_id_now,
                                                  'session': session_now['session'][0],
                                                  'trial': trialnum,
                                                  'trial_note_type': 'autowater',
                                                  'trial_note': 'left and right'
                                                  }
                                        elif any((df_behavior_trial['MSG'] == 'Auto_Water_L') & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                                            trialnotedata = {
                                                  'subject_id': subject_id_now,
                                                  'session': session_now['session'][0],
                                                  'trial': trialnum,
                                                  'trial_note_type': 'autowater',
                                                  'trial_note': 'left'
                                                  }
                                        elif any((df_behavior_trial['MSG'] == 'Auto_Water_R') & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                                            trialnotedata = {
                                                  'subject_id': subject_id_now,
                                                  'session': session_now['session'][0],
                                                  'trial': trialnum,
                                                  'trial_note_type': 'autowater',
                                                  'trial_note': 'right'
                                                  }
                                        if trialnotedata:
                                            experiment.TrialNote().insert1(trialnotedata, allow_direct_insert=True)
                                        
                                        #% add Go Cue
                                        GoCueTimes = (time_GoCue.to_numpy() - time_TrialStart.to_datetime64())/np.timedelta64(1,'s')
                                        trialeventdatas = list()
                                        for trial_event_i,trial_event_time  in enumerate(GoCueTimes):
                                            trialeventdatas.append({
                                                    'subject_id': subject_id_now,
                                                    'session': session_now['session'][0],
                                                    'trial': trialnum,
                                                    'trial_event_id': trial_event_i,
                                                    'trial_event_type': 'go',
                                                    'trial_event_time': trial_event_time,
                                                    'duration': 0, #for a go cue
                                                    })
                                        experiment.TrialEvent().insert(trialeventdatas, allow_direct_insert=True)
                                        #% add licks
                                        actioneventdatas = list()
                                        LeftLickTimes = (time_lick_left.to_numpy() - time_TrialStart.to_datetime64())/np.timedelta64(1,'s')
                                        RightLickTimes = (time_lick_right.to_numpy() - time_TrialStart.to_datetime64())/np.timedelta64(1,'s')
                                        for action_event_time in LeftLickTimes:
                                            actioneventdatas.append({
                                                    'subject_id': subject_id_now,
                                                    'session': session_now['session'][0],
                                                    'trial': trialnum,
                                                    'action_event_id':len(actioneventdatas)+1,
                                                    'action_event_time':action_event_time,
                                                    'action_event_type': 'left lick'
                                                    })
                                        for action_event_time in RightLickTimes:
                                            actioneventdatas.append({
                                                    'subject_id': subject_id_now,
                                                    'session': session_now['session'][0],
                                                    'trial': trialnum,
                                                    'action_event_id':len(actioneventdatas)+1,
                                                    'action_event_time':action_event_time,
                                                    'action_event_type': 'right lick'
                                                    })
                                        if actioneventdatas:
                                            experiment.ActionEvent().insert(actioneventdatas, allow_direct_insert=True)