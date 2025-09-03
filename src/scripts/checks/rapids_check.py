import os
import cudf as cd

base = os.getcwd()
sett = base + r'/src/scripts/db_dummy/sett-scenario1.csv'

settings = cd.read_csv(sett)
print(settings)