import pandas as pd

from wildfire.data.firms import load_all_firms
from wildfire.processing.confidence import add_unified_confidence


df = add_unified_confidence(load_all_firms(country="Spain", years=[2023, 2024]))
print(df)
print(df.columns)

df_no_og = df.drop(columns=["confidence_og_num", "confidence_og_cat"])
print(df)

print(df_no_og.isnull().sum())

# test