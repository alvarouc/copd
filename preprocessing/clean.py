import pandas as pd
from myutils import make_logger

logger = make_logger('clean')

if __name__ == "__main__":

    df = pd.read_pickle('merged.pkl')

    logger.info('Merged shape {}'.format(df.shape))
    
    # Remove rows without birth date
    df = df.loc[df.PT_BIRTH_DT.notnull(),:]
    logger.info('Have birth date {}'.format(df.shape))
    
    # Computing age as lab date - date of birth
    df['AGE'] = df['LAB_DT'] - df['PT_BIRTH_DT'].dt.year
    df = df[df.AGE>10]

    logger.info('Have birth date {}'.format(df.shape))

    df.to_csv('clean.csv', index=False)
