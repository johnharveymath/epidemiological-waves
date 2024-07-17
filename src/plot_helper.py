import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from wavefinder.wavelist import WaveList
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def plot_results(raw_data: DataFrame, peaks_before: DataFrame, peaks_after: DataFrame):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex='all')
    # plot peaks-trough pairs from sub_a
    ax0.set_title('Data before Algorithm')
    ax0.plot(raw_data.values)
    ax0.scatter(peaks_before['location'].values,
                raw_data.values[peaks_before['location'].values.astype(int)], color='red', marker='o')
    # plot peaks-trough pairs from sub_b
    ax1.set_title('Data after Algorithm')
    ax1.plot(raw_data.values)
    ax1.scatter(peaks_after['location'].values,
                raw_data.values[peaks_after['location'].values.astype(int)], color='red', marker='o')
    plt.show()

def plot_final_peaks(results: DataFrame, cases: DataFrame, patient_list: DataFrame, country_code: str, country_name: str,
                         plot_path: str):
    """
    Plots how additional peaks are imputed in input_wavelist from reference_wavelist by WaveCrossValidator

    Parameters:
        results (DataFrame): The peaks and troughs found in the input_wavelist after cross-validation.
        cases (DataFrame): The original data.
        filename (str): The filename to save the plot.
        plot_path (str): The path to save the plot.
    """

    fig, axs = plt.subplots(nrows=1, ncols=1)

    origin_date = cases.date.values[0]

    # plot peaks
    axs.set_title(f'Waves in {country_name} and patient visits')
    axs.plot(cases.new_per_day_smooth.values)
    axs.scatter(results['location'].values, cases.new_per_day_smooth[results['location'].values.astype(int)].values, color='red', marker='o')

    # plot patients
    patient_data = patient_list if country_name=="Total" else patient_list[patient_list['Country']==country_name]
    visits = (patient_data.visit_date - origin_date).dt.days.values
    constant = [50] * len(visits)
    axs.scatter(visits, constant, color='black', marker='x')

    # format x-axix
    def todate(x, pos, today=origin_date):
        return today + dt.timedelta(days=x)

    fmt = ticker.FuncFormatter(todate)
    axs.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, country_code + '_cross_validated.png'))
    plt.close('all')
