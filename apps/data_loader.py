import numpy as np
import pandas as pd
import os


def load_data(path):

    x_df = pd.read_csv(os.path.join(path, 'x.csv'))
    y_df = pd.read_csv(os.path.join(path, 'y.csv'))
    
    x = x_df.to_numpy()
    y = y_df.to_numpy()   

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y