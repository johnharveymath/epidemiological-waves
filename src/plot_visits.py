import os
from tqdm import tqdm
from epidemicwaveclassifier import EpidemicWaveClassifier
from data_provider import DataProvider
from config import Config
from waveanalysispanel import WaveAnalysisPanel
from figures import Figures

import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from wavefinder.wavelist import WaveList
from datetime import datetime
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


config = Config(os.path.dirname(os.path.realpath(__file__)))

# get patient list
patient_list = pd.read_csv(os.path.join(config.data_path, 'patient_list.csv'))
patient_list.replace(to_replace='Czech Republic', value='Czechia', inplace=True)
patient_list['date_covid_onset'] = pd.to_datetime(patient_list['date_covid_onset'], format='%d/%m/%Y').dt.date
patient_list['visit_date'] = pd.to_datetime(patient_list['visit_date'], format='%d/%m/%Y').dt.date

# get waves
waves = pd.read_csv(os.path.join(config.data_path, 'final_waves.csv'))

# assign wave to each visit
patient_list['wave'] = 5 # gives grey colour on plot, good choice for uncategorised
for country_code in waves['countrycode']:
    country_name = waves.loc[waves['countrycode'] == country_code, 'country'].values[0]
    if country_name in ['Total', 'Brazil', 'Egypt', 'United Kingdom', 'India', 'Chile']:
        continue
    for wave in reversed(range(1,5)):
        wave_end = waves.loc[waves['country'] == country_name][f'Wave{wave}End'].values[0]
        wave_end = datetime.strptime(wave_end, '%d/%m/%Y').date()
        patient_list.loc[(patient_list.Country == country_name) & (patient_list.visit_date < wave_end), 'wave'] = wave

    #patient_list.query('Country == @country_name and visit_date < @wave_1_end')['wave']=1

# set up plot
fig, axs = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(10,10)
origin_date = dt.date(2020,1,1)
axs.set_title(f'Patient visits in each country')

# place the countries in correct order
countries = waves['country'].iloc[::-1]
dummy, = plt.plot([0]*(len(countries)-1),countries[0:-1])
dummy.remove()

# plot patients
patient_data = patient_list
visits = (patient_data.visit_date - origin_date).dt.days.values
axs.scatter(visits, patient_data.Country, cmap='Set1', c=patient_data.wave, marker='x')

# format x-axix
def todate(x, pos, today=origin_date):
    return today + dt.timedelta(days=x)

fmt = ticker.FuncFormatter(todate)
axs.xaxis.set_major_formatter(fmt)
fig.autofmt_xdate(rotation=45)

# finalise and save
#fig.tight_layout()
plt.savefig(os.path.join(config.plot_path, '..', 'visits', 'visits.png'), bbox_inches='tight')
plt.close('all')