import os
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import wkt
from tqdm import tqdm
from config import Config
from data_provider import DataProvider
from epidemicwaveclassifier import EpidemicWaveClassifier


class Figures:
    def __init__(self, config: Config, epi_panel: pd.core.frame.DataFrame,
                 data_provider: DataProvider, epi_classifier: EpidemicWaveClassifier ):
        self.data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data')
        self.plot_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'plots')
        self.config = config
        self.data_provider = data_provider
        self.epi_panel = epi_panel
        self.epi_classifier = epi_classifier
        return

    def _figure_0(self):
        countries = ['ZMB','GBR','GHA','CRI']
        figure_0_series = self.data_provider.epidemiology_series.loc[
            self.data_provider.epidemiology_series['countrycode'].isin(countries),
            ['country', 'countrycode', 'date', 'new_per_day',
             'new_per_day_smooth', 'dead_per_day', 'dead_per_day_smooth']]
        figure_0_panel = self.epi_panel.loc[self.epi_panel['countrycode'].isin(countries),
                                            ['class', 'country', 'countrycode', 'population', 't0_10_dead',
                                             'date_peak_1', 'peak_1', 'wave_start_1', 'wave_end_1',
                                             'date_peak_2', 'peak_2', 'wave_start_2', 'wave_end_2',
                                             'date_peak_3', 'peak_3', 'wave_start_3', 'wave_end_3']]
        figure_0_series.to_csv(os.path.join(self.data_dir, 'figure_0_series.csv'))
        figure_0_panel.to_csv(os.path.join(self.data_dir, 'figure_0_panel.csv'))
        return

    def _figure_2(self):
        countries = ['ITA', 'FRA', 'USA', 'ZMB','GBR','GHA','CRI']
        figure_2 = self.data_provider.epidemiology_series.loc[
            self.data_provider.epidemiology_series['countrycode'].isin(countries),
            ['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
             'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smooth',
             'positive_rate', 'positive_rate_smooth', 'cfr_smooth']]
        figure_2.to_csv(os.path.join(self.data_dir, 'figure_2.csv'))
        return

    def _figure_deaths(self):
        out = pd.DataFrame(columns=['countrycode', 'date', 'location', 'dead_per_day_smooth'])
        for k, v in self.epi_classifier.deaths_summary_output.items():
            if len(v) == 0:
                continue
            for i, wave in enumerate(v):
                if wave['peak_ind'] == 0:
                    continue
                upsert = {}
                upsert['countrycode'] = k
                upsert['date'] = wave['date']
                upsert['location'] = wave['location']
                upsert['dead_per_day_smooth'] = wave['y_position']
                out = out.append(upsert, ignore_index=True)
            continue
        out.to_csv(os.path.join(self.data_dir, 'figure_dead_all.csv'))
        return

    def main(self):
        self._figure_2()
        self._figure_deaths()
        return
