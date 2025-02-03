import pandas as pd 
from epiweeks import Week
import matplotlib.pyplot as plt 

states_ne = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'SE', 'RN']
states_se = ['SP', 'RJ', 'ES', 'MG']
states_sul = ['RS', 'SC', 'PR' ]
states_ce = ['DF', 'MT', 'MS', 'GO']
states_no = ['AP', 'TO', 'RR', 'RO', 'AM' ,'AC', 'PA']

states_BR = states_ne+states_se+states_no+states_ce+states_sul

dates_23 = pd.date_range(start= Week(2022, 41).startdate().strftime('%Y-%m-%d'),
              end= Week(2023, 39).startdate().strftime('%Y-%m-%d'),
              freq='W-SUN')

dates_24 = pd.date_range(start= Week(2023, 41).startdate().strftime('%Y-%m-%d'),
              end= Week(2024, 23).startdate().strftime('%Y-%m-%d'),
              freq='W-SUN')

dates_25 = pd.date_range(start= Week(2024, 41).startdate().strftime('%Y-%m-%d'),
              end= Week(2025, 40).startdate().strftime('%Y-%m-%d'),
              freq='W-SUN')

UNIQUE_MODELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'ln_base', 'ln_crps', 'ln_log', 'lin_crps', 'lin_log']
# Define colors manually, mapping each region to a color from tab10
colors = plt.get_cmap('Set3').colors[:len(UNIQUE_MODELS)]
COLOR_MAP = dict(zip(UNIQUE_MODELS, colors))

def format_data(state, models_by_state, data_all, df_preds_all): 
    '''
    Function to organize the predictions and data available by season
    '''
    models = sorted(models_by_state.loc[models_by_state.state == state]['model_id'].values[0])
        
    data = data_all.loc[data_all.uf == state].reset_index(drop = True)
    
    data_23 = data.loc[data.date.isin(dates_23)]
    data_24 = data.loc[data.date.isin(dates_24)]
    
    preds = df_preds_all.loc[df_preds_all.state == state]
        
    preds.date = pd.to_datetime(preds.date)
    
    preds_23 = preds.loc[preds.date.isin(dates_23)]
    preds_23 = preds_23.drop_duplicates(subset=['date', 'model_id'], keep='first').reset_index(drop=True)
        
    preds_24 = preds.loc[preds.date.isin(dates_24)]
    preds_24 = preds_24.drop_duplicates(subset=['date', 'model_id'], keep='first').reset_index(drop=True)
    preds_25 = preds.loc[preds.date.isin(dates_25)]
    preds_25 = preds_25.drop_duplicates(subset=['date', 'model_id'], keep='first').reset_index(drop=True)


    return data_23, data_24, preds_23, preds_24, preds_25, models


def load_preds(exclude = True, rename = True):
    '''
    Function to load the predictions and actual data to make de figures 
    '''
    # get the predictions from the separated models: 
    df_preds_all = pd.read_csv('../predictions/preds_models.csv.gz', index_col = 'Unnamed: 0')
    df_preds_all = df_preds_all.rename(columns = {'adm_1': 'state'})
    df_preds_all['model_id'] = df_preds_all['model_id'].replace({26:25})
    # df_preds_all = df_preds_all.loc[(df_preds_all.model_id != 21) & (df_preds_all.model_id != 25)]
    df_preds_all.model_id = df_preds_all.model_id.astype(int)

    df_preds_all = df_preds_all.sort_values(by = 'model_id')
    
    if exclude: 
    # excluindo linhas com previsões com alguma inconsistência: 
        df_preds_problem = df_preds_all.loc[df_preds_all.pred <= 0.1]
        
        excluded_models = list(set(zip(df_preds_problem.model_id, df_preds_problem.state)))
        
        df_preds_all = df_preds_all[~df_preds_all.apply(lambda row: (row['model_id'], row['state']) in excluded_models, axis=1)]
    
    models_by_state = df_preds_all.groupby('state')['model_id'].agg(lambda x: sorted(list(set(x)))).reset_index()

    df_preds_all.date = pd.to_datetime(df_preds_all.date)

    df_preds_all['model_id'] = df_preds_all['model_id'].astype(str)
    
    # get the predictions from the ensemble baseline:
    df_base = pd.read_csv('../predictions/ensemble_2023_2024_base_log_normal.csv')
    df_base.date = pd.to_datetime(df_base.date)
    df_base['model_id'] = 'ln_base'
    
    # get the in sample preds using CRPS and Log Score for 2023
    
    df_crps = pd.read_csv('../predictions/ensemble_2023_2024_E2.csv')
    df_crps.date = pd.to_datetime(df_crps.date)
    df_crps['model_id'] = 'ln_crps'
    
    df_ls = pd.read_csv('../predictions/ensemble_2023_2024_log_score_log_normal.csv')
    df_ls.date = pd.to_datetime(df_ls.date)
    df_ls['model_id'] = 'ln_log'

    df_crps_mix = pd.read_csv('../predictions/ensemble_2023_2024_crps_lin_ln.csv')
    df_crps_mix['model_id'] = 'lin_crps'

    df_log_mix = pd.read_csv('../predictions/ensemble_2023_2024_log_score_lin_ln.csv')
    df_log_mix['model_id'] = 'lin_log'

    df_preds = pd.concat([df_preds_all, df_base, df_crps, df_ls, df_crps_mix, df_log_mix])
    df_preds.date = pd.to_datetime(df_preds.date)

    if rename: 
        replace_models_name  ={'21':'M1', '22':'M2',
                            '25':'M3', '26':'M3', '27':'M4',
                            '28':'M5', '30':'M6',
                            '34':'M7'}
        
        df_preds['model_id'] = df_preds['model_id'].replace(replace_models_name)

    data_all = pd.read_csv('../data/dengue_uf.csv.gz')
    data_all.date = pd.to_datetime(data_all.date)

    return df_preds, models_by_state, data_all