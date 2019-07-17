import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# use creds to create a client to interact with the Google Drive API
scope = ['https://www.googleapis.com/auth/analytics.readonly',
      'https://www.googleapis.com/auth/drive',
      'https://www.googleapis.com/auth/spreadsheets',
      ]#['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

#%% open 
def fetch_animal_metadata():
    wb = client.open("Surgery, water restriction and training")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    idx_main = sheetnames.index('Surgery')
    main_sheet = wb.get_worksheet(idx_main)
    df = pd.DataFrame(main_sheet.get_all_records())
    return df

def fetch_water_restriction_metadata(ID):
    wb = client.open("Surgery, water restriction and training")
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    idx_now = sheetnames.index(ID)
    if idx_now > -1:
        sheet_now = wb.get_worksheet(idx_now)
        temp = dict()
        header = sheet_now.row_values(1)
        for i,head in enumerate(header):
            temp[head]=sheet_now.col_values(i+1)[1:]
        df = pd.DataFrame.from_dict(temp, orient='index')
        df.transpose()
        df = df.transpose()
        return df
    else:
        return None

def fetch_lastmodify_time_animal_metadata():
    modifiedtime = None
    ID = None
    service = build('drive', 'v3', credentials=creds)
    wb = client.open("Surgery, water restriction and training")
    ID = wb.id
    if ID:
        modifiedtime = service.files().get(fileId = ID,fields = 'modifiedTime').execute()
    return modifiedtime

#%%


    
    
#%%
# =============================================================================
# #%%
# 
# 
# # Find a workbook by name and open the first sheet
# # Make sure you use the right name here.
# wb = client.open("waterport calibration")
# sheet = wb.get_worksheet(0)
# # Extract and print all of the values
# df = pd.DataFrame(sheet.get_all_records())
# df = df[(df['num']!='Average') & (df['num']!= 'STD')]
# Left = np.mean(df[df['Direction']=='Left'])
# Right = np.mean(df[df['Direction']=='Right'])
# for vol in range(10,60,10):
#     
#     print(vol)
# plt.plot(Left)
# plt.plot(Right)
# =============================================================================


