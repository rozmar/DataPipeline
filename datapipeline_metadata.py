import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
import json
import time as timer
#% connect to server
import datajoint as dj
dj.conn()
dj.config['project'] = 'foraging'
from pipeline import pipeline_tools
from pipeline import lab, experiment
#%%
def populatemetadata():
    #%% save metadata from google drive if necessairy
    lastmodify = online_notebook.fetch_lastmodify_time_animal_metadata()
    with open(dj.config['locations.metadata_behavior']+'last_modify_time.json') as timedata:
        lastmodify_prev = json.loads(timedata.read())
    if lastmodify != lastmodify_prev:
        print('updating surgery and WR metadata from google drive')
        dj.config['locations.metadata_behavior']
        df_surgery = online_notebook.fetch_animal_metadata()
        df_surgery.to_csv(dj.config['locations.metadata_behavior']+'Surgery.csv')
        IDs = df_surgery['ID'].tolist()
        for ID in IDs:
            df_wr = online_notebook.fetch_water_restriction_metadata(ID)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(dj.config['locations.metadata_behavior']+ID+'.csv') 
        with open(dj.config['locations.metadata_behavior']+'last_modify_time.json', "w") as write_file:
            json.dump(lastmodify, write_file)
        print('surgery and WR metadata updated')
    
    lastmodify = online_notebook.fetch_lastmodify_time_lab_metadata()
    with open(dj.config['locations.metadata_lab']+'last_modify_time.json') as timedata:
        lastmodify_prev = json.loads(timedata.read())
    if lastmodify != lastmodify_prev:
        print('updating Lab metadata from google drive')
        dj.config['locations.metadata_lab']
        IDs = ['Experimenter','Rig','Virus']
        for ID in IDs:
            df_wr = online_notebook.fetch_lab_metadata(ID)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(dj.config['locations.metadata_lab']+ID+'.csv') 

        with open(dj.config['locations.metadata_lab']+'last_modify_time.json', "w") as write_file:
            json.dump(lastmodify, write_file)
        print('Lab metadata updated')
    
    #%% add users
    df_experimenters = pd.read_csv(dj.config['locations.metadata_lab']+'Experimenter.csv')
    experimenterdata = list()
    for experimenter in df_experimenters.iterrows():
        experimenter = experimenter[1]
        dictnow = {'username':experimenter['username'],'fullname':experimenter['fullname']}
        experimenterdata.append(dictnow)
    print('adding experimenters')
    for experimenternow in experimenterdata:
        try:
            lab.Person().insert1(experimenternow)
        except dj.DuplicateError:
            print('duplicate. experimenter: ',experimenternow['username'], ' already exists')
    
    #%% add rigs
    df_rigs = pd.read_csv(dj.config['locations.metadata_lab']+'Rig.csv')
    rigdata = list()
    for rig in df_rigs.iterrows():
        rig = rig[1]
        dictnow = {'rig':rig['rig'],'room':rig['room'],'rig_description':rig['rig_description']}
        rigdata.append(dictnow)
    print('adding rigs')
    for rignow in rigdata:
        try:
            lab.Rig().insert1(rignow)
        except dj.DuplicateError:
            print('duplicate. rig: ',rignow['rig'], ' already exists')
            
    #%% add viruses
    df_viruses = pd.read_csv(dj.config['locations.metadata_lab']+'Virus.csv')
    virusdata = list()
    serotypedata = list()
    for virus in df_viruses.iterrows():
        virus = virus[1]
        if type(virus['remarks']) != str:
            virus['remarks'] = ''
        dictnow = {'virus_id':virus['virus_id'],
                   'virus_source':virus['virus_source'],
                   'serotype':virus['serotype'],
                   'username':virus['username'],
                   'virus_name':virus['virus_name'],
                   'titer':virus['titer'],
                   'order_date':virus['order_date'],
                   'remarks':virus['remarks']}
        virusdata.append(dictnow)
        dictnow = {'serotype':virus['serotype']}
        serotypedata.append(dictnow)
    print('adding rigs')
    for virusnow,serotypenow in zip(virusdata,serotypedata):
        try:
            lab.Serotype().insert1(serotypenow)
        except dj.DuplicateError:
            print('duplicate serotype: ',serotypenow['serotype'], ' already exists')
        try:
            lab.Virus().insert1(virusnow)
        except dj.DuplicateError:
            print('duplicate virus: ',virusnow['virus_name'], ' already exists')
    #%% populate subjects, surgeries and water restrictions
    print('adding surgeries and stuff')
    df_surgery = pd.read_csv(dj.config['locations.metadata_behavior']+'Surgery.csv')
    #%%
    for item in df_surgery.iterrows():
        if item[1]['project'] == dj.config['project'] and (item[1]['status'] == 'training' or item[1]['status'] == 'sacrificed'):
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
                try:
                    lab.Surgery().insert1(surgerydata)
                except dj.DuplicateError:
                    print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ', start_time)
                #checking craniotomies
                #%
                cranioidx = 1
                while 'craniotomy diameter ('+str(cranioidx)+')' in item[1].keys() and item[1]['craniotomy diameter ('+str(cranioidx)+')'] and (type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == int or type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == float):
                    if item[1]['craniotomy surgery id ('+str(cranioidx)+')'] == surgeryidx:
                        proceduredata = {
                                'surgery_id': surgeryidx,
                                'subject_id':item[1]['animal#'],
                                'procedure_id':cranioidx,
                                'skull_reference':item[1]['craniotomy reference ('+str(cranioidx)+')'],
                                'ml_location':item[1]['craniotomy lateral ('+str(cranioidx)+')'],
                                'ap_location':item[1]['craniotomy anterior ('+str(cranioidx)+')'],
                                'surgery_procedure_description': 'craniotomy: ' + item[1]['craniotomy comments ('+str(cranioidx)+')'],
                                }
                        try:
                            lab.Surgery.Procedure().insert1(proceduredata)
                        except dj.DuplicateError:
                            print('duplicate cranio for animal ',item[1]['animal#'], ' already exists: ', cranioidx)
                    cranioidx += 1
                #% 
                
                virusinjidx = 1
                while 'virus inj surgery id ('+str(virusinjidx)+')' in item[1].keys() and item[1]['virus inj virus id ('+str(virusinjidx)+')'] and item[1]['virus inj surgery id ('+str(virusinjidx)+')']:
                    if item[1]['virus inj surgery id ('+str(virusinjidx)+')'] == surgeryidx:
    # =============================================================================
    #                     print('waiting')
    #                     timer.sleep(1000)
    # =============================================================================
                        if '[' in item[1]['virus inj lateral ('+str(virusinjidx)+')']:
                            virus_ml_locations = eval(item[1]['virus inj lateral ('+str(virusinjidx)+')'])
                            virus_ap_locations = eval(item[1]['virus inj anterior ('+str(virusinjidx)+')'])
                            virus_dv_locations = eval(item[1]['virus inj ventral ('+str(virusinjidx)+')'])
                            virus_volumes = eval(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])
                        else:
                            virus_ml_locations = [int(item[1]['virus inj lateral ('+str(virusinjidx)+')'])]
                            virus_ap_locations = [int(item[1]['virus inj anterior ('+str(virusinjidx)+')'])]
                            virus_dv_locations = [int(item[1]['virus inj ventral ('+str(virusinjidx)+')'])]
                            virus_volumes = [int(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])]
                            
                        for virus_ml_location,virus_ap_location,virus_dv_location,virus_volume in zip(virus_ml_locations,virus_ap_locations,virus_dv_locations,virus_volumes):
                            injidx = len(lab.Surgery.VirusInjection() & surgerydata) +1
                            virusinjdata = {
                                    'surgery_id': surgeryidx,
                                    'subject_id':item[1]['animal#'],
                                    'injection_id':injidx,
                                    'virus_id':item[1]['virus inj virus id ('+str(virusinjidx)+')'],
                                    'skull_reference':item[1]['virus inj reference ('+str(virusinjidx)+')'],
                                    'ml_location':virus_ml_location,
                                    'ap_location':virus_ap_location,
                                    'dv_location':virus_dv_location,
                                    'volume':virus_volume,
                                    'dilution':item[1]['virus inj dilution ('+str(virusinjidx)+')'],
                                    'description': 'virus injection: ' + item[1]['virus inj comments ('+str(virusinjidx)+')'],
                                    }
                            try:
                                lab.Surgery.VirusInjection().insert1(virusinjdata)
                            except dj.DuplicateError:
                                print('duplicate virus injection for animal ',item[1]['animal#'], ' already exists: ', injidx)
                    virusinjidx += 1    
                #%
                
                surgeryidx += 1
                    
                #%
            if item[1]['ID']:
                #df_wr = online_notebook.fetch_water_restriction_metadata(item[1]['ID'])
                try:
                    df_wr = pd.read_csv(dj.config['locations.metadata_behavior']+item[1]['ID']+'.csv')
                except:
                    df_wr = None
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
                      