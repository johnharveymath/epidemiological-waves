import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config


def parse_variant_sheet(sheet_name, raw_df):

    variant = sheet_name

    dates = pd.to_datetime(
        raw_df.iloc[0],
        errors='coerce'
    )
    date_cols = dates[dates.notna()].index

    rows = []
    r = 1

    while r < len(raw_df) - 2:

        country = raw_df.iloc[r, 0]

        if pd.isna(country):
            r += 1
            continue

        country = str(country).strip()

        count_row = r
        total_row = r + 1

        count_label = str(
            raw_df.iloc[count_row, 1]
        ).strip()

        total_label = str(
            raw_df.iloc[total_row, 1]
        ).strip()

        if count_label != 'count':
            raise ValueError(
                f'{sheet_name}: expected \'count\' after '
                f'{country}, found {count_label!r}'
            )

        if total_label != 'total':
            raise ValueError(
                f'{sheet_name}: expected \'total\' after '
                f'{country}, found {total_label!r}'
            )

        for c in date_cols:
            rows.append(
                {
                    'country': country,
                    'week': dates[c],
                    'variant': variant,
                    'count': (
                        0
                        if pd.isna(raw_df.iloc[count_row, c])
                        else float(raw_df.iloc[count_row, c])
                    ),
                    'total': (
                        np.nan
                        if pd.isna(raw_df.iloc[total_row, c])
                        else float(raw_df.iloc[total_row, c])
                    ),
                }
            )

        r += 2

    return pd.DataFrame(rows)

def excel_import(filename):
    xls = pd.ExcelFile(filename)

    variant_sheets = [
        s for s in xls.sheet_names
        if s != 'README'
    ]

    dfs = []

    for sheet in variant_sheets:
        #raw = pd.read_excel(
        #    filename,
        #    sheet_name=sheet,
        #    header=None
        #)
        raw = xls.parse(sheet, header=None)

        dfs.append(
            parse_variant_sheet(sheet, raw)
        )

    df = pd.concat(dfs, ignore_index=True)
    return df

def build_out_df(df):
    all_countries = sorted(df['country'].unique())
    all_weeks = sorted(df['week'].unique())
    all_variants = sorted(df['variant'].unique())

    full_index = pd.MultiIndex.from_product(
        [
            all_countries,
            all_weeks,
            all_variants
        ],
        names=[
            'country',
            'week',
            'variant'
        ]
    )

    df = (
        df
        .set_index(['country', 'week', 'variant'])
        .reindex(full_index)
    )

    df['count'] = df['count'].fillna(0)

    group_total = (
        df['total']
        .groupby(['country', 'week'])
        .transform('max')
    )

    df['total'] = df['total'].fillna(group_total)
    df['total'] = df['total'].fillna(1)

    return df


def add_not_recorded(df):
    known = (
        df.groupby(['country', 'week'])['count']
        .sum()
    )
    totals = (
        df.groupby(['country', 'week'])['total']
        .first()
    )
    not_recorded = (
            totals - known
    ).clip(lower=0)
    nr = pd.DataFrame(
        {
            'count': not_recorded,
            'total': totals
        }
    )

    nr['variant'] = 'Not recorded'

    nr = (
        nr.reset_index()
        .set_index(
            ['country', 'week', 'variant']
        )
    )
    df = pd.concat([df, nr])
    return df



def build_dominance_table(
    df,
    dominance_threshold,
):

    freq = (
        df.reset_index()
          .pivot_table(
              index=['country', 'week'],
              columns='variant',
              values='frequency',
              fill_value=0,
          )
    )

    result = freq.assign(
        top_variant=freq.idxmax(axis=1),
        top_frequency=freq.max(axis=1),
    )

    result['dominant_variant'] = result['top_variant'].where(
        result['top_frequency'] >= dominance_threshold
    )

    return result.reset_index()[['country', 'week', 'dominant_variant']]

def get_dominance_periods(dominance):
    d = (
        dominance
        .sort_values(['country', 'week'])
        .copy()
    )

    d['block'] = (
        d.groupby('country')['dominant_variant']
        .transform(
            lambda s: (
                s.ne(s.shift())
            ).cumsum()
        )
    )

    periods = (
        d.dropna(subset=['dominant_variant'])
        .groupby(
            [
                'country',
                'dominant_variant',
                'block',
            ]
        )
        .agg(
            start=('week', 'min'),
            end=('week', 'max'),
            weeks=('week', 'count'),
        )
        .reset_index()
    )

    return periods



def plot_country(
        country,
        df,
        periods,
        variants_of_interest,
        country_visits,
        plot_threshold = 0.1,
):

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={
            'height_ratios': [4, 1]
        }
    )

    country_df = (
        df.loc[country]
        .reset_index()
    )

    plot_df = (
        country_df[
            country_df['variant'].isin(
                variants_of_interest
            )
        ]
        .pivot(
            index='week',
            columns='variant',
            values='frequency'
        )
    )

    plot_df = (
        country_df
        .pivot(
            index='week',
            columns='variant',
            values='frequency'
        )
    )

    plot_df = plot_df.loc[
              :,
              (plot_df.max() > plot_threshold)
              ]

    PLOT_ORDER = [
        'Not recorded',
        'VOC Alpha',
        'VOC Gamma',
        'VOC Delta',
        'VOC Omicron',
    ]

    priority = [v for v in PLOT_ORDER if v in plot_df.columns]

    other_variants = sorted(
        [
            v for v in plot_df.columns
            if v not in PLOT_ORDER
        ]
    )

    plot_df = plot_df[priority + other_variants]

    VARIANT_COLORS = {
        'Not recorded': '#9467bd',  # purple
        'VOC Alpha': '#1f77b4',  # blue
        'VOC Gamma': '#2ca02c',  # green
        'VOC Delta': '#ff7f0e',  # orange
        'VOC Omicron': '#d62728',  # red
    }

    used_colors = set(VARIANT_COLORS.values())

    default_colors = (
        plt.rcParams['axes.prop_cycle']
        .by_key()['color']
    )

    available_colors = [
        c for c in default_colors
        if c not in used_colors
    ]

    for i, variant in enumerate(other_variants):

        if variant not in VARIANT_COLORS:
            VARIANT_COLORS[variant] = available_colors[
                i % len(available_colors)
                ]

    plot_colors = [
        VARIANT_COLORS[v]
        for v in plot_df.columns
    ]

    ax1.stackplot(
        plot_df.index,
        plot_df.T,
        labels=plot_df.columns,
        colors=plot_colors,
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Variant frequency')
    ax1.set_title(country)

    ax1.grid(axis='y', alpha=0.3)

    ax1.legend(
        title='Variant',
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
    )

    # dominance periods

    periods_country = periods[
        periods['country'] == country
        ]

    y = 0

    for _, row in periods_country.iterrows():
        ax2.broken_barh(
            [(row['start'], row['end'] - row['start'])],
            (-0.25, 0.5),
            facecolors=VARIANT_COLORS[row['dominant_variant']],
        )

    ax2.scatter(
        country_visits['visit_date'],
        np.zeros(len(country_visits)),
        marker='x',
        s=40,
        c='black',
        linewidths=1.5,
        zorder=10,
    )

    ax2.set_yticks([])
    ax2.set_ylabel('Dominance')

    plt.tight_layout()

    return fig



config = Config(os.path.dirname(os.path.realpath(__file__)))
filename = os.path.join(config.data_path, 'gisaid_variants_statistics_2025_01_19_1130.xlsx')

try:
    df = pd.read_pickle(os.path.join(config.data_path, 'gisaid_tidy_df.pkl'))
except:
    df = excel_import(filename)
    df = build_out_df(df)
    df = add_not_recorded(df)
    df['frequency'] = (
        df['count'] / df['total']
    )
    df.to_pickle(os.path.join(config.data_path, 'gisaid_tidy_df.pkl'))

weeks = df.index.get_level_values('week')
df = df[
(weeks >= '2020-01-01')
&
(weeks <= '2022-12-31')
]

dominance = build_dominance_table(
    df,
    dominance_threshold=0.75
)

periods = get_dominance_periods(dominance)

variant_list = ['Not recorded', 'VOC Alpha', 'VOC Beta', 'VOC Gamma', 'VOC Delta', 'VOC Omicron']

# get patient list
patient_list = pd.read_csv(os.path.join(config.data_path, 'patient_list.csv'))

energy_countries = patient_list['Country'].unique()
gisaid_countries = sorted(df.index.get_level_values('country').unique())

for country in energy_countries:
    if country not in gisaid_countries:
        raise ValueError(f'{country} not found in GISAID data')

patient_list['date_covid_onset'] = pd.to_datetime(patient_list['date_covid_onset'], format='%d/%m/%Y').dt.date
patient_list['visit_date'] = pd.to_datetime(patient_list['visit_date'], format='%d/%m/%Y').dt.date

for country in energy_countries:

    country_visits = patient_list[
        patient_list['Country'] == country
        ]

    fig = plot_country(
        country,
        df,
        periods,
        variant_list,
        country_visits
    )

    fig.savefig(
        os.path.join(config.plot_path, '..', 'variants', f'{country}_variants.png'),
        dpi=300,
        bbox_inches='tight',
    )

    plt.close(fig)
