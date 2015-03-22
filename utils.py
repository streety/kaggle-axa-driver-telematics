import numpy as np
import os
import pandas as pd
import random

driver_directory = '/home/jonathan/dev/kaggle-drivers/drivers/'
features_directory = '/home/jonathan/dev/kaggle-drivers/features/'

num_drivers = 3612
num_trips = 200


def scale_probabilities(p, threshold, norm_threshold):
    #print(p, threshold, norm_threshold)
    result = np.zeros(p.shape)
    result[p <= threshold] = p[p <= threshold] * norm_threshold / threshold
    result[p > threshold] = 1 - ((1 - p[p > threshold]) * \
            (1 - norm_threshold) / (1 - threshold))
    return result


def get_folders():
    return [i for i in \
            next(os.walk(driver_directory))[1]]


def get_files(folder):
    return [os.path.join(folder, i) for i in \
            next(os.walk(driver_directory))[2]]


def import_data(filename):
    return pd.read_csv(filename, sep=',', header=0)


def get_features_for_driver(driver):
    trips = []
    for i in range(1, 201):
        trips.append(get_features_for_trip(driver, i))
    return pd.concat(trips, axis=1).T


def get_features_for_trip(driver, trip):
    features = pd.read_csv(os.path.join(features_directory,
        '{0}-{1}.csv'.format(driver, trip)), sep=' ', index_col=0)
    features = pd.concat([features, 
        pd.DataFrame({'driver': int(driver), 'trip': int(trip)}, index=['x']).T])
    features = features.fillna(0)
    features.columns = ['{0}_{1}'.format(driver, trip)]
    return features


def get_random_trip_collection(n, exclude):
    trips = []
    drivers = get_folders()
    random.shuffle(drivers)
    for i in range(n):
        trips.append(get_features_for_trip(drivers[i], 
            random.randint(1,num_trips)))
    return pd.concat(trips, axis=1).T


def build_dataset(driver, falsely_belong, not_belong):
    driver_features = get_features_for_driver(driver)
    driver_features['class'] = 1
    falsely_belong = falsely_belong.copy()
    falsely_belong['class'] = 1
    falsely_belong['falsely_belong'] = 1
    not_belong = not_belong.copy()
    not_belong['class'] = 0
    dataset = pd.concat([driver_features, falsely_belong, not_belong], axis=0)
    dataset = dataset.fillna(0)
    return dataset
