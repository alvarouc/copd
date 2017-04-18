import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from myutils import make_logger, describe

logger = make_logger('merge')

def read_comorb(data_dir):
    logger.info('Reading co-morbities table')

    df = pd.read_csv(data_dir + 'COPD_COMORBID_CBM.txt', sep='\t')
    # DATE parsing
    dt_columns = df.filter(like='_DT').columns
    df[dt_columns] = df[dt_columns].apply(
        lambda x: pd.to_datetime(x, errors='coerce'))
    df = df.set_index('PT_ID')\
           .filter(like='_DT')\
           .apply(lambda x: x.dt.year)\
           .reset_index()

    return df


def read_demographics(data_dir):
    logger.info('Reading demographics')

    df = pd.read_csv(data_dir + 'COPD_INDEX_TABLE_CBM.txt', sep='\t')
    # DATE parsing
    dt_columns = df.filter(like='_DT').columns
    df[dt_columns] = df[dt_columns].apply(
        lambda x: pd.to_datetime(x, errors='coerce'))

    df = df.loc[:, ['PT_ID', 'PT_BIRTH_DT', 'PT_DEATH_DT', 'PT_SEX',
                    'PT_STATUS', 'PT_RACE_1', 'PT_HX_SMOKE']]

    # Remove death dates before birth date
    logger.warning('Patients with death date before birth \n {}'.format(
        df[df.PT_DEATH_DT < df.PT_BIRTH_DT]))

    logger.info('Demograhics: {}'.format(df.shape))

    return df


def read_clinical(data_dir):
    logger.info('Reading Clinical data')
    df = pd.read_csv(data_dir + 'COPD_CLINICAL_DATA_CM.txt', sep='\t')
    # DATE parsing
    df['LAB_DT'] = pd.to_datetime(df['LAB_DT'], format='%d%b%Y:%H:%M:%S.%f')\
                     .astype('datetime64[ns]')
    df.drop(['PT_CURRENT_AGE', 'PT_SEX', 'PT_HX_SMOKE', 'INDEX_DT',
             'LAB_COMP_CD', 'DLCO_UNCORRECTED', 'BMI', 'DLVA', 'VA',
             'HDL', 'LDL', 'TRIGLYCERIDES'],
            axis=1, inplace=True)
    # Uniformize lab tests to DLCO, DLVA, HDL, LDL, TRYGLY,
    df.loc[['LDL' in val for val in df['LAB_COMP_NM']], 'LAB_COMP_NM'] = 'LDL'
    df.loc[['HDL' in val for val in df['LAB_COMP_NM']], 'LAB_COMP_NM'] = 'HDL'
    df.loc[['TRIGLY' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'TRIGLYCERIDES'

    # Summariing lab data by year: median
    df = df.groupby([df.PT_ID, df.LAB_DT.dt.year, df.LAB_COMP_NM])\
           .median()['LAB_RES_VAL_NUM'].unstack('LAB_COMP_NM')
    df.reset_index(inplace=True)

    logger.info('Clinical: {}'.format(df.shape))

    return df


def read_metabolic(data_dir):
    logger.info('Reading Metabolic data')

    df = pd.read_csv(data_dir + 'COPD_MET_PANEL_CM.txt', sep='\t')
    df['LAB_DT'] = pd.to_datetime(
        df['LAB_DT'], format='%d%b%Y:%H:%M:%S.%f').astype('datetime64[ns]')
    df = df[['PT_ID', 'LAB_DT', 'LAB_COMP_NM',
             'LAB_RES_VAL_NUM', 'LAB_RES_UNIT']]

    # Uniformize lab names
    df.loc[['WBC' in val for val in df['LAB_COMP_NM']], 'LAB_COMP_NM'] = 'WBC'
    df.loc[['BILIRUBIN' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'BILIRUBIN'
    df.loc[['CREATININE' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'CREATININE'
    df.loc[['POTASSIUM' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'POTASSIUM'
    df.loc[['PROTEIN' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'PROTEIN'
    df.loc[['CO2' in val for val in df['LAB_COMP_NM']], 'LAB_COMP_NM'] = 'CO2'
    df.loc[['ALK' in val for val in df['LAB_COMP_NM']], 'LAB_COMP_NM'] = 'ALK'
    df.loc[['CHLORIDE' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'CHLORIDE'
    df.loc[['GLUCOSE' in val for val in df['LAB_COMP_NM']],
           'LAB_COMP_NM'] = 'GLUCOSE'

    df = df[[name in ['BUN', 'WBC', 'GLUCOSE', 'BILIRUBIN', 'CREATININE',
                      'POTASSIUM', 'PROTEIN', 'CALCIUM', 'CO2', 'ALK',
                      'CHLORIDE', 'SODIUM', 'ANION GAP']
             for name in df['LAB_COMP_NM']]]

    df = df[(df['LAB_COMP_NM'] != 'WBC') | (df['LAB_RES_VAL_NUM'] < 50)]
    df = df.groupby([df.PT_ID, df.LAB_DT.dt.year, df.LAB_COMP_NM])\
           .median().unstack('LAB_COMP_NM')

    df.columns = df.columns.droplevel()
    df.reset_index(inplace=True)

    logger.info('Metabolic: {}'.format(df.shape))

    return df


def read_vitals(data_dir):
    logger.info('Reading Vitals data')

    df = pd.read_csv(data_dir + 'COPD ClusteringCOPD_VITALS_CM.txt', sep='\t')
    df['VTL_DT'] = pd.to_datetime(
        df['VTL_DT'], format='%d%b%Y:%H:%M:%S.%f').astype('datetime64[ns]')
    df = df[['PT_ID', 'VTL_DT', 'VTL_TYPE', 'VTL_VALUE']]
    df.columns = ['PT_ID', 'LAB_DT', 'LAB_COMP_NM', 'VTL_VALUE']
    df = df.groupby([df.PT_ID, df.LAB_DT.dt.year, df.LAB_COMP_NM]).median()[
        'VTL_VALUE'].unstack('LAB_COMP_NM')
    df.reset_index(inplace=True)
    logger.info('Vitals: {}'.format(df.shape))
    return df


if __name__ == "__main__":
    data_dir = '/home/aeulloacerna/data/copd/data/'

    demo = read_demographics(data_dir)
    clinical = read_clinical(data_dir)
    met = read_metabolic(data_dir)
    vitals = read_vitals(data_dir)
    comob = read_comorb(data_dir)
    comob_list = comob.columns.drop('PT_ID').values
    
    logger.info('Merging')
    merged = demo.merge(clinical,on='PT_ID', how='inner')\
                 .merge(met, on=['PT_ID','LAB_DT'], how='outer')\
                 .merge(vitals, on=['PT_ID','LAB_DT'], how='outer')\
                 .merge(comob, on='PT_ID')
    logger.info('Parsing co-comorbidities by date')
    merged[comob_list] = merged[comob_list]\
                         .apply(lambda x: x<=merged.LAB_DT)

    logger.info('Saving merged.pkl {}'.format(merged.shape))
    merged.to_csv('merged.csv', index=False)
    merged.to_pickle('merged.pkl')
