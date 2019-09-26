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
from pipeline import lab, experiment,ephys_patch
from pipeline import behavioranal, ephysanal
import ray
#%%
@ray.remote
def populatemytables_core_paralel(arguments,runround):
    if runround == 1:
        behavioranal.TrialReactionTime().populate(**arguments)
        behavioranal.SessionReactionTimeHistogram().populate(**arguments)
        behavioranal.SessionLickRhythmHistogram().populate(**arguments)  
        behavioranal.SessionTrainingType().populate(**arguments)  
#        behavioranal.SessionRewardRatio().populate(**arguments)  
        behavioranal.BlockRewardRatio().populate(**arguments)  
        behavioranal.BlockChoiceRatio().populate(**arguments)  
        behavioranal.BlockAutoWaterCount().populate(**arguments)  
        behavioranal.SessionBlockSwitchChoices().populate(**arguments)  
        behavioranal.SessionFittedChoiceCoefficients().populate(**arguments)
        behavioranal.SubjectFittedChoiceCoefficients.populate(**arguments)
        behavioranal.SubjectFittedChoiceCoefficientsOnlyRewards.populate(**arguments)    
    if runround == 2:
        behavioranal.SessionPsychometricCurveDataBoxCar.populate(**arguments)
        behavioranal.SessionPsychometricCurveDataFitted.populate(**arguments)
    if runround == 3:
        behavioranal.SubjectPsychometricCurveBoxCar.populate(**arguments)
        behavioranal.SubjectPsychometricCurveFitted.populate(**arguments)

def populatemytables_core(arguments,runround):
    if runround == 1:
        behavioranal.TrialReactionTime().populate(**arguments)
        behavioranal.SessionReactionTimeHistogram().populate(**arguments)
        behavioranal.SessionLickRhythmHistogram().populate(**arguments)  
        behavioranal.SessionTrainingType().populate(**arguments)  
#        behavioranal.SessionRewardRatio().populate(**arguments)  
        behavioranal.BlockRewardRatio().populate(**arguments)  
        behavioranal.BlockChoiceRatio().populate(**arguments)  
        behavioranal.BlockAutoWaterCount().populate(**arguments)  
        behavioranal.SessionBlockSwitchChoices().populate(**arguments)  
        behavioranal.SessionFittedChoiceCoefficients().populate(**arguments)
        behavioranal.SubjectFittedChoiceCoefficients.populate(**arguments)
        behavioranal.SubjectFittedChoiceCoefficientsOnlyRewards.populate(**arguments)    
    if runround == 2:
        behavioranal.SessionPsychometricCurveDataBoxCar.populate(**arguments)
        behavioranal.SessionPsychometricCurveDataFitted.populate(**arguments)
    if runround == 3:
        behavioranal.SubjectPsychometricCurveBoxCar.populate(**arguments)
        behavioranal.SubjectPsychometricCurveFitted.populate(**arguments)
        
def populatemytables(paralel = True, cores = 6):
    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
    #df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
    for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects      and removing subject related analysis   
       # if df_surgery['status'][df_surgery['ID']==subject_now].values[0] != 'sacrificed': # only if the animal is still in training..
        if len((experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"').fetch('session')) > 0:
            
            schemas_todel = [behavioranal.SubjectFittedChoiceCoefficients() & 'subject_id = "' + str(subject_id_now)+'"',
                             behavioranal.SubjectFittedChoiceCoefficientsOnlyRewards() & 'subject_id = "' + str(subject_id_now)+'"',
                             behavioranal.SubjectPsychometricCurveBoxCar() & 'subject_id = "' + str(subject_id_now)+'"',
                             behavioranal.SessionPsychometricCurveDataFitted() & 'subject_id = "' + str(subject_id_now)+'"',
                             behavioranal.SubjectPsychometricCurveFitted() & 'subject_id = "' + str(subject_id_now)+'"',
                             ]
            dj.config['safemode'] = False
            for schema_todel in schemas_todel:
                schema_todel.delete()
            dj.config['safemode'] = True       
    if paralel:
        
        ray.init()
        for runround in [1,2,3]:
            arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
            print('round '+str(runround)+' of populate')
            result_ids = []
            for coreidx in range(cores):
                result_ids.append(populatemytables_core_paralel.remote(arguments,runround))        
            ray.get(result_ids)
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)
            
        ray.shutdown()
    else:
        arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
        populatemytables_core(arguments)

def populatebehavior(paralel = True,drop_last_session_for_mice_in_training = True):
    print('adding behavior experiments')
    if paralel:
        ray.init()
        result_ids = []
        #%%
        IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
        df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
        for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects      and removing last session      
            if drop_last_session_for_mice_in_training == True and df_surgery['status'][df_surgery['ID']==subject_now].values[0] != 'sacrificed': # the last session is deleted only if the animal is still in training..
                print(df_surgery['status'][df_surgery['ID']==subject_now].values[0])
                if len((experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"').fetch('session')) > 0:
                    sessiontodel = np.max((experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"').fetch('session'))
                    session_todel = experiment.Session() & 'subject_id = "' + str(subject_id_now)+'"' & 'session = ' + str(sessiontodel)
                    dj.config['safemode'] = False
                    session_todel.delete()
                    dj.config['safemode'] = True   
                    #%%
        for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects                       
            dict_now = dict()
            dict_now[subject_now] = subject_id_now
            result_ids.append(populatebehavior_core.remote(dict_now))    
        resultnottouse = ray.get(result_ids)
        ray.shutdown()
    else:
        populatebehavior_core(drop_last_session_for_mice_in_training = drop_last_session_for_mice_in_training)


# =============================================================================
# IDs = None
# =============================================================================
@ray.remote
def populatebehavior_core(IDs = None):
    if IDs:
        print('subject started:')
        print(IDs.keys())
        print(IDs.values())
    directories = dict()
    directories = {'behavior_project_dirs' : ['/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging_again',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-2/Foraging_homecage',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-3/Foraging_homecage',
                                              '/home/rozmar/Data/Behavior/Behavior_room/Tower-1/Foraging',]
        }
    projects = list()
    for projectdir in directories['behavior_project_dirs']:
        projects.append(Project())
        projects[-1].load(projectdir)
    #df_surgery = pd.read_csv(dj.config['locations.metadata']+'Surgery.csv')
    if IDs == None:
        IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}
    #%%
    for subject_now,subject_id_now in zip(IDs.keys(),IDs.values()): # iterating over subjects
        print('subject: ',subject_now)
    # =============================================================================
    #         if drop_last_session_for_mice_in_training:
    #             delete_last_session_before_upload = True
    #         else:
    #             delete_last_session_before_upload = False
    #         #df_wr = online_notebook.fetch_water_restriction_metadata(subject_now)
    # =============================================================================
        df_wr = pd.read_csv(dj.config['locations.metadata']+subject_now+'.csv')
        for df_wr_row in df_wr.iterrows():
            if df_wr_row[1]['Time'] and type(df_wr_row[1]['Time'])==str and df_wr_row[1]['Time-end'] and type(df_wr_row[1]['Time-end'])==str and df_wr_row[1]['Training type'] != 'restriction' and df_wr_row[1]['Training type'] != 'handling': # we use it when both start and end times are filled in, restriction and handling is skipped
                #%%
                date_now = df_wr_row[1].Date.replace('-','')
                print('subject: ',subject_now,'  date: ',date_now)
    
                sessions_now = list()
                session_start_times_now = list()
                experimentnames_now = list()
                for proj in projects: #
                    exps = proj.experiments
                    for exp in exps:
                        stps = exp.setups
                        for stp in stps:
                            #sessions = stp.sessions
                            for session in stp.sessions:
                                if session.subjects and session.subjects[0].find(subject_now) >-1 and session.name.startswith(date_now):
                                    sessions_now.append(session)
                                    session_start_times_now.append(session.started)
                                    experimentnames_now.append(exp.name)
                order = np.argsort(session_start_times_now)
                #%%
                session_date = datetime(int(date_now[0:4]),int(date_now[4:6]),int(date_now[6:8]))
                if len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date > "'+str(session_date)+'"') != 0: # if it is not the last
                    print('session already imported, skipping: ' + str(session_date))
                    dotheupload = False
                elif len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"') != 0: # if it is the last
                    dotheupload = False
    # =============================================================================
    #                     if delete_last_session_before_upload == True and df_surgery['status'][df_surgery['ID']==subject_now].values[0] != 'sacrificed': # the last session is deleted only if the animal is still in training..
    #                         print(df_surgery['status'][df_surgery['ID']==subject_now].values[0])
    #                         if len(experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"') != 0:
    #                             print('dropping last session')
    #                             session_todel =experiment.Session() & 'subject_id = "'+str(subject_id_now)+'"' & 'session_date = "'+str(session_date)+'"'
    #                             dj.config['safemode'] = False
    #                             session_todel.delete()
    #                             dj.config['safemode'] = True
    #                         delete_last_session_before_upload = False
    #                         dotheupload = True
    #                     else:
    #                         dotheupload = False
    # =============================================================================
                else: # reuploading new session that is not present on the server
                    dotheupload = True
                #%%
                for session_idx in order:
                    session = sessions_now[session_idx]
                    experiment_name = experimentnames_now[session_idx]
                    csvfilename = (Path(session.path) / (Path(session.path).name + '.csv'))
                    if dotheupload:
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
                            elif session.setup_name.lower() in ['tower-1']:
                                setupname = 'Training-Tower-1'
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
                            #print(date_now + ' - ' + session.task_name)
                            session_now = (experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"').fetch()
                            session_start_time = datetime.combine(session_now['session_date'][0],datetime.min.time()) +session_now['session_time'][0]
                            #% extracting trial data
                            trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
                            trial_start_idxs = pd.Index([0]).append(trial_start_idxs[1:]) # so the random seed will be present
                            trial_end_idxs = trial_start_idxs[1:].append(pd.Index([(max(df_behavior_session.index))]))
                            #trial_end_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'END-TRIAL')].index
                            prevtrialstarttime = np.nan
                            blocknum_local_prev = np.nan
                            for trial_start_idx,trial_end_idx in zip(trial_start_idxs,trial_end_idxs) :
                                df_behavior_trial = df_behavior_session[trial_start_idx:trial_end_idx+1]
                                #Trials without GoCue  are skipped
                                if len(df_behavior_trial['PC-TIME'][(df_behavior_trial['MSG'] == 'GoCue') & (df_behavior_trial['TYPE'] == 'TRANSITION')]) > 0:
                
                                    trial_start_time = df_behavior_session['PC-TIME'][trial_start_idx].to_pydatetime() - session_start_time
                                    trial_stop_time = df_behavior_session['PC-TIME'][trial_end_idx].to_pydatetime() - session_start_time
                                    trial_start_time_decimal = decimal.Decimal(trial_start_time.total_seconds()).quantize(decimal.Decimal('.0001'))
                                    if len(experiment.SessionTrial() & 'session =' + str(session_now['session'][0]) & 'trial_start_time ="' + str(trial_start_time_decimal) + '"') == 0 and trial_start_time != prevtrialstarttime:# importing if this trial is not already imported
                                        prevtrialstarttime = trial_start_time
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
                                        if 'Block_number' in df_behavior_session.columns and np.isnan(df_behavior_trial['Block_number'].to_list()[0]):
                                            if np.isnan(blocknum_local_prev):
                                                blocknum_local = 0
                                            else:
                                                blocknum_local = blocknum_local_prev
                                        elif 'Block_number' in df_behavior_session.columns:
                                            blocknum_local = int(df_behavior_trial['Block_number'].to_list()[0])-1
                                            blocknum_local_prev  = blocknum_local
                                            
                                        if 'Block_number' in df_behavior_session.columns:# and not np.isnan(df_behavior_trial['Block_number'].to_list()[0]):
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
                                                if blocknum>100:
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
                                                #print('new block added: ' + str (block_start_time))
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
                                            elif trialnum == 1: # if can't inherit, we guess a protocol..
                                                task_protocol = 12
                                        elif experiment_name in ['Delayed_foraging','Delayed foraging','Foraging_homecage','Foraging']:
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
                                            elif trialnum == 1: # if can't inherit, we guess a protocol..
                                                task_protocol = 15
                                        else:
                                            #%
                                            task = np.nan
                                            task_protocol = 'nan' 
                                            print('task name not handled:'+experiment_name)
                                        #%%
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
                                        #%% add autowater
                                        if any((df_behavior_trial['TYPE'] == 'STATE') & (df_behavior_trial['MSG'] == 'Auto_Water_L')) or any((df_behavior_trial['TYPE'] == 'STATE') &(df_behavior_trial['MSG'] == 'Auto_Water_R')):
                                            Lidx = (df_behavior_trial['TYPE'] == 'STATE') & (df_behavior_trial['MSG'] == 'Auto_Water_L')
                                            Lidx = Lidx.idxmax()
                                            Ridx = (df_behavior_trial['TYPE'] == 'STATE') & (df_behavior_trial['MSG'] == 'Auto_Water_R')
                                            Ridx = Ridx.idxmax()
                                            #%%
                                            if float(df_behavior_trial['+INFO'][Ridx])>.001 and float(df_behavior_trial['+INFO'][Lidx])>.001:
                                                trialnotedata = {
                                                        'subject_id': subject_id_now,
                                                        'session': session_now['session'][0],
                                                        'trial': trialnum,
                                                        'trial_note_type': 'autowater',
                                                        'trial_note': 'left and right'
                                                        }
                                            elif float(df_behavior_trial['+INFO'][Lidx])>.001:
                                                trialnotedata = {
                                                        'subject_id': subject_id_now,
                                                        'session': session_now['session'][0],
                                                        'trial': trialnum,
                                                        'trial_note_type': 'autowater',
                                                        'trial_note': 'left'
                                                        }
                                            elif float(df_behavior_trial['+INFO'][Ridx])>.001:
                                                trialnotedata = {
                                                        'subject_id': subject_id_now,
                                                        'session': session_now['session'][0],
                                                        'trial': trialnum,
                                                        'trial_note_type': 'autowater',
                                                        'trial_note': 'right'
                                                        }
                                            else:
                                                trialnotedata = None # autowater was on but there was no water due to the probabilities
    
                                        if trialnotedata:
                                            experiment.TrialNote().insert1(trialnotedata, allow_direct_insert=True)
                                        #%% add random seed start
                                        trialnotedata = None
                                        if any(df_behavior_trial['MSG'] == 'Random seed:'):
                                            seedidx = (df_behavior_trial['MSG'] == 'Random seed:').idxmax() + 1
                                            trialnotedata = {
                                                        'subject_id': subject_id_now,
                                                        'session': session_now['session'][0],
                                                        'trial': trialnum,
                                                        'trial_note_type': 'random_seed_start',
                                                        'trial_note': str(df_behavior_trial['MSG'][seedidx])
                                                        }
                                            experiment.TrialNote().insert1(trialnotedata, allow_direct_insert=True)
                                        #%% add watervalve data
                                        if 'var_motor:LickPort_Lateral_pos' in df_behavior_trial.keys():
                                            water_valve_lateral_pos = df_behavior_trial['var_motor:LickPort_Lateral_pos'].values[0]
                                        else:
                                            water_valve_lateral_pos = 0
                                        if 'var_motor:LickPort_RostroCaudal_pos' in df_behavior_trial.keys():
                                            water_valve_rostrocaudal_pos = df_behavior_trial['var_motor:LickPort_RostroCaudal_pos'].values[0]
                                        else:
                                            water_valve_rostrocaudal_pos = 0
                                        water_valve_dorsoventral_pos = 0
                                        if 'var:ValveOpenTime_L' in df_behavior_trial.keys():
                                            water_valve_time_left = df_behavior_trial['var:ValveOpenTime_L'].values[0]
                                        else:
                                            water_valve_time_left = -1
                                        if 'var:ValveOpenTime_R' in df_behavior_trial.keys():
                                            water_valve_time_right = df_behavior_trial['var:ValveOpenTime_R'].values[0]
                                        else:
                                            water_valve_time_right = -1
                                        watervalvedata ={
                                                'subject_id': subject_id_now,
                                                'session': session_now['session'][0],
                                                'trial': trialnum,
                                                'water_valve_lateral_pos': water_valve_lateral_pos,
                                                'water_valve_rostrocaudal_pos': water_valve_rostrocaudal_pos,
                                                'water_valve_dorsoventral_pos': water_valve_dorsoventral_pos,
                                                'water_valve_time_left': water_valve_time_left,
                                                'water_valve_time_right': water_valve_time_right,
                                                }
                                        experiment.WaterValveData().insert1(watervalvedata, allow_direct_insert=True)
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