import pandas as pd

def clean_cab_data(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()

    # converting and droping timestamp column
    if 'datetime' in df.columns and 'timestamp' in df.columns:
        df.drop('timestamp',axis=1,inplace=True)
        df['datetime']=pd.to_datetime(df['datetime'],errors='coerce')
    elif 'timestamp' in df.columns:
        df['datetime']=pd.to_datetime(df['timestamp'],unit='s',errors='coerce',utc=False) 
        df.drop('timestamp',axis=1,inplace=True)

    # droping duplicates
    df.drop_duplicates(inplace=True)

    # finding any columns with have "Time " in it and finding if its nuerical using pd.api.types.is_numeric_dtype if it is than convert it into my datetime 
    for col in df.columns:
        if "Time" in col and pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col]=pd.to_datetime(df[col],unit='s',errors='coerce',utc=False)
            except Exception as e:
                print(f"Skipping column {col}:{e}")

    for col in df.columns:
        if "summary" in col:
            df[col]=df[col].str.strip()           
                                
    return df
         