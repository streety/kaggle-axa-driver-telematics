import numpy as np
import os
import pandas as pd
import random
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import time
from joblib import Parallel, delayed


from skimage import exposure

import utils

columns = ['speed_5_pcnt', 'speed_10_pcnt', 'speed_15_pcnt', 'speed_20_pcnt', 'speed_25_pcnt', 'speed_30_pcnt', 'speed_35_pcnt', 'speed_40_pcnt', 'speed_45_pcnt', 'speed_50_pcnt', 'speed_55_pcnt', 'speed_60_pcnt', 'speed_65_pcnt', 'speed_70_pcnt', 'speed_75_pcnt', 'speed_80_pcnt', 'speed_85_pcnt', 'speed_90_pcnt', 'speed_95_pcnt', 'speed_100_pcnt', 'tang_accel_5_pcnt', 'tang_accel_10_pcnt', 'tang_accel_15_pcnt', 'tang_accel_20_pcnt', 'tang_accel_25_pcnt', 'tang_accel_30_pcnt', 'tang_accel_35_pcnt', 'tang_accel_40_pcnt', 'tang_accel_45_pcnt', 'tang_accel_50_pcnt', 'tang_accel_55_pcnt', 'tang_accel_60_pcnt', 'tang_accel_65_pcnt', 'tang_accel_70_pcnt', 'tang_accel_75_pcnt', 'tang_accel_80_pcnt', 'tang_accel_85_pcnt', 'tang_accel_90_pcnt', 'tang_accel_95_pcnt', 'tang_accel_100_pcnt', 'norm_accel_5_pcnt', 'norm_accel_10_pcnt', 'norm_accel_15_pcnt', 'norm_accel_20_pcnt', 'norm_accel_25_pcnt', 'norm_accel_30_pcnt', 'norm_accel_35_pcnt', 'norm_accel_40_pcnt', 'norm_accel_45_pcnt', 'norm_accel_50_pcnt', 'norm_accel_55_pcnt', 'norm_accel_60_pcnt', 'norm_accel_65_pcnt', 'norm_accel_70_pcnt', 'norm_accel_75_pcnt', 'norm_accel_80_pcnt', 'norm_accel_85_pcnt', 'norm_accel_90_pcnt', 'norm_accel_95_pcnt', 'norm_accel_100_pcnt', 'total_accel_5_pcnt', 'total_accel_10_pcnt', 'total_accel_15_pcnt', 'total_accel_20_pcnt', 'total_accel_25_pcnt', 'total_accel_30_pcnt', 'total_accel_35_pcnt', 'total_accel_40_pcnt', 'total_accel_45_pcnt', 'total_accel_50_pcnt', 'total_accel_55_pcnt', 'total_accel_60_pcnt', 'total_accel_65_pcnt', 'total_accel_70_pcnt', 'total_accel_75_pcnt', 'total_accel_80_pcnt', 'total_accel_85_pcnt', 'total_accel_90_pcnt', 'total_accel_95_pcnt', 'total_accel_100_pcnt', 'cur_5_pcnt', 'cur_10_pcnt', 'cur_15_pcnt', 'cur_20_pcnt', 'cur_25_pcnt', 'cur_30_pcnt', 'cur_35_pcnt', 'cur_40_pcnt', 'cur_45_pcnt', 'cur_50_pcnt', 'cur_55_pcnt', 'cur_60_pcnt', 'cur_65_pcnt', 'cur_70_pcnt', 'cur_75_pcnt', 'cur_80_pcnt', 'cur_85_pcnt', 'cur_90_pcnt', 'cur_95_pcnt', 'cur_100_pcnt', 'distance']

from sklearn.base import TransformerMixin


class ModelTransformer(TransformerMixin):
    """Wrap a classifier model so that it can be used in a pipeline"""
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        #print(X.shape)
        return self.model.predict_proba(X)

    def predict_proba(self, X, **transform_params):
        return self.transform(X, **transform_params)


def build_pipeline():
    pipeline = Pipeline([('classifier', 
        ModelTransformer(RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=1,
        random_state=0)))])
    return pipeline

def generate_predictions_for_driver(driver, falsely_belong, not_belong, 
        pipeline, norm_threshold, columns):
    dataset = utils.build_dataset(driver, falsely_belong, not_belong)
    dataset.to_csv('test.csv')
    X = dataset[columns]
    y = dataset['class']
    clf = pipeline.fit(X, y)
    predictions = clf.predict_proba(X)[:,1]
    #print(predictions.shape)
    equalize_predictions = exposure.equalize_hist(predictions)
    threshold = equalize_predictions[(dataset['falsely_belong'] == 1).values]. \
            mean()
    scaled_predictions = utils.scale_probabilities(equalize_predictions,
                                            threshold,
                                            norm_threshold)
    return (pd.DataFrame({'predictions':predictions,
                    'equalize_predictions':equalize_predictions,
                    'scaled_predictions':scaled_predictions}, 
                    index=dataset.index), threshold)
    return (predictions, equalize_predictions, scaled_predictions, threshold)


def build_models_for_all_drivers(columns, norm_threshold=0.5, 
        num_falsely_belong=10, num_not_belong=500, store_filename='store.h5'):
    drivers = utils.get_folders()
    falsely_belong = utils.get_random_trip_collection(num_falsely_belong, 
            1000000).fillna(0)
    not_belong = utils.get_random_trip_collection(num_not_belong, 
            1000000).fillna(0)
    random.shuffle(drivers)
    #print(falsely_belong.dtypes)
    #print(not_belong.dtypes)
    store = pd.HDFStore(store_filename)
    store['falsely_belong'] = falsely_belong
    store['not_belong'] = not_belong
    print(len(drivers))
    for i, driver in enumerate(drivers):
        print(i, driver)
        pipeline = build_pipeline()
        predictions, threshold = generate_predictions_for_driver(driver, 
                falsely_belong,
                not_belong,
                pipeline,
                norm_threshold,
                columns)
        store.put('predictions_{0}'.format(driver), predictions)
    store.close()



def run(seed=None):
    if not seed:
        seed = int(time.time())
    print(seed)
    random.seed(seed)
    store_filename = 'model_RF500_{0}_store.h5'.format(seed)
    features = utils.get_features_for_trip(1, 1)
    columns = list(features.T.columns)
    columns.remove('driver')
    columns.remove('trip')
    build_models_for_all_drivers(columns, store_filename=store_filename)

if __name__ == '__main__':
    seed = int(time.time())
    Parallel(n_jobs=8)(delayed(run)('{0}-{1}'.format(seed, i)) for i in range(16))
