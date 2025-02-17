
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return 'datasets/MSL_SMAP'
        elif database == 'power':
            return 'CARLA/datasets/Power'
        elif database == 'yahoo':
            return 'datasets/yahoo'
        elif database == 'smd':
            return 'CARLA/datasets/SMD'
        elif database == 'swat':
            return 'CARLA/datasets/SWAT'
        elif database == 'wadi':
            return 'CARLA/datasets/WADI'
        elif database == 'kpi':
            return 'CARLA/datasets/KPI'
        
        else:
            raise NotImplementedError
