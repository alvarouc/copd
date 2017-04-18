import pandas as pd
import multiprocessing as mp
from myutils import make_logger
logger = make_logger('impute')


def interp(grouped):
    # iterating through groups return (name, group)
    return grouped[1]\
        .set_index('LAB_DT')\
        .interpolate(method='index')\
        .fillna(method='bfill')\
        .fillna(method='pad')\
        .reset_index()


if __name__ == "__main__":

    df = pd.read_pickle('clean.pkl')

    icols = df.select_dtypes(include=['float64']).columns

    logger.info('Starting imputation {:,} missing values.'.format(
        df[icols].isnull().values.sum()))

    gcol = icols.tolist()
    gcol.extend(['PT_ID', 'LAB_DT'])
    with mp.Pool(8) as pool:
        temp = pool.map(interp, df[gcol].groupby('PT_ID'))
    temp = pd.concat(temp)
    df.set_index(['PT_ID', 'LAB_DT'], inplace=True)
    df.loc[:, icols] = temp.set_index(
        ['PT_ID', 'LAB_DT'])[icols].values
    df.reset_index(inplace=True)
    logger.info('Starting imputation {:,} missing values.'.format(
        df[icols].isnull().values.sum()))

    toim = df.select_dtypes(include=['float64', 'bool']).columns
    df[toim].to_csv('to_mice.csv', index=False)
