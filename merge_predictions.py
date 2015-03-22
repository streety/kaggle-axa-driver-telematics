import pandas as pd
import os
import re
import numpy as np

directory = '/home/jonathan/dev/kaggle-drivers/'

if __name__ == '__main__':
    regex = re.compile('model_RF500_\d+-\d+_store\.h5')
    files = [i for i in os.listdir(directory) if i.endswith('.h5')]
    #print(files)
    files = [i for i in files if regex.match(i)]
    files.sort()
    print(files)
    #exit()
    predictions_all_models = []
    for f in files:
        with pd.HDFStore(os.path.join(directory, f)) as store:
            keys = store.keys()
            keys = [i for i in keys if i.startswith('/predictions')]
            predictions = []
            for k in keys:
                predictions.append(store[k].iloc[:200,0])
            predictions_df = pd.concat(predictions)
            predictions_all_models.append(predictions_df)
    predictions_all_models_df = pd.concat(predictions_all_models, axis=1)
    predictions_all_models_df.mean(axis=1).to_csv('submission.csv')
