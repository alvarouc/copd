import pandas as pd
import numpy as np
from myutils import make_logger

logger = make_logger('clean')

if __name__ == "__main__":

    df = pd.read_pickle('merged.pkl')

    logger.info('Merged shape {}'.format(df.shape))

    # Remove rows without birth date
    df = df.loc[df.PT_BIRTH_DT.notnull(), :]
    logger.info('Have birth date {}'.format(df.shape))

    # Computing age as lab date - date of birth
    df['AGE'] = df['LAB_DT'] - df['PT_BIRTH_DT'].dt.year
    df = df[df.AGE > 18]

    logger.info('Older than 18 {}'.format(df.shape))

    temp = df.select_dtypes(include=['float64']).drop('PT_HX_SMOKE', axis=1)
    temp[temp > (temp.mean() + 3 * temp.std())] = np.nan
    temp[temp < 0] = np.nan
    df = df[temp.notnull().sum(axis=1) > 1]
    logger.info('Have more than one measurement {}'.format(df.shape))

    n_outliers = temp.apply(lambda x: (x > (x.mean() + 3 * x.std())).sum(),
                            axis=0)

    temp = df.groupby('PT_ID').agg({'LAB_DT': np.max,
                                    'PT_DEATH_DT': np.max})
    black_list = temp[temp['PT_DEATH_DT'].dt.year < temp.LAB_DT].index.values
    df = df[[id not in black_list for id in df['PT_ID']]]
    logger.info('Do not have LAB after death date {}'.format(df.shape))

    df = df[df.PT_STATUS != 'UNKNOWN']
    logger.info('Known whether alive or dead {}'.format(df.shape))

    df.to_pickle('clean.pkl')
    df.to_csv('clean.csv', index=False)
