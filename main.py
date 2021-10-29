import data_gen.data_main
import model_main
from data_gen.var_lib import *
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    get_data = data_gen.data_main.final_program()
    get_data.main()
    get_model = model_main.model_gen(pd.read_pickle(pre_split_path))
    get_model.main()

