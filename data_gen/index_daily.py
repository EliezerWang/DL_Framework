import tushare as ts
import pandas as pd
from data_gen.var_lib import *

pro = ts.pro_api()
df = pro.index_daily(ts_code=code[0])
