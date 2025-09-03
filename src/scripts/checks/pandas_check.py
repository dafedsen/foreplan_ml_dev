import os
import pandas as pd

base = os.getcwd()
sett = base + r'/src/scripts/db_dummy/sett-scenario1.csv'

settings = pd.read_csv(sett)
print(settings)