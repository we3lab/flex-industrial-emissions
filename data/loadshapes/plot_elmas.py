import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

elmas_data = pd.read_csv('data/loadshapes/elmas/Time_series_18_clusters.csv', sep=';', decimal=',', parse_dates=[0], index_col=0)
cluster_names_elmas = json.load(open('data/loadshapes/elmas/cluster_names.json'))


def process_time_period(data, time_period, day_period):
    """
    Retrieve the subset of the data for the given month 
    and return the mean load for each hour.
    """
    period_data = data.loc[data.index.month == (1 if time_period == 'January' else 7)]
    weekday_data = period_data.loc[period_data.index.dayofweek < day_period]
    grouped = weekday_data.groupby(weekday_data.index.hour)
    return pd.DataFrame({'mean': grouped.mean(), 'std': grouped.std()})

def plot_load_shapes(data, cluster_names, plot_type, day_period=5, num_clusters = 18):
    """
    Plot the load shapes for the ELMAS data, cluster names, 
    plot type (original or normalized), and day period (5 or 7).
    """
    fig, axs = plt.subplots(nrows=num_clusters // 3, ncols=3, figsize=(18, 36))
    axs = axs.flatten()
    max_load = data.max().max()

    for i in range(num_clusters):
        cluster_data = data.iloc[:, i]
        
        january_data = process_time_period(cluster_data, 'January', day_period)
        july_data = process_time_period(cluster_data, 'July', day_period)
        
        if plot_type == 'normalized':
            january_mean = january_data['mean'] / january_data['mean'].mean()
            january_std = january_data['std'] / january_data['mean'].mean()
            july_mean = july_data['mean'] / july_data['mean'].mean()
            july_std = july_data['std'] / july_data['mean'].mean()
            ylim = (0.0, 2.5)
        else:
            january_mean = january_data['mean']
            january_std = january_data['std']
            july_mean = july_data['mean']
            july_std = july_data['std']
            ylim = (0, max_load)
        
        axs[i].plot(january_mean.index, january_mean, label='January', color='blue')
        axs[i].fill_between(january_mean.index, january_mean - january_std, january_mean + january_std, alpha=0.2, color='blue')
        axs[i].plot(july_mean.index, july_mean, label='July', color='red')
        axs[i].fill_between(july_mean.index, july_mean - july_std, july_mean + july_std, alpha=0.2, color='red')
        axs[i].set_title(f'{i + 1}: {cluster_names[str(i + 1)]}')
        axs[i].set_xlabel('Hour')
        axs[i].set_ylabel('Load')
        axs[i].set_ylim(ylim)
        axs[i].legend()
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(12))

    fig.suptitle(f'{"Normalized" if plot_type == "normalized" else "Original"} Load Shapes: January vs July', fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig(f'data/loadshapes/elmas/{plot_type}.png')

plot_load_shapes(elmas_data, cluster_names_elmas, 'original')
plot_load_shapes(elmas_data, cluster_names_elmas, 'normalized')


selected_clusters = range(1, 19)

january_df = pd.DataFrame()
for cluster in selected_clusters:
    cluster_data = elmas_data.iloc[:, cluster - 1]
    month_data = cluster_data[cluster_data.index.month == 1]
    january_df[cluster] = month_data.groupby(month_data.index.hour).mean()
january_df.to_csv('../data/loadshapes/elmas_data_january_24hr_profile.csv')

# Process and save July data
july_df = pd.DataFrame()
for cluster in selected_clusters:
    cluster_data = elmas_data.iloc[:, cluster - 1]
    month_data = cluster_data[cluster_data.index.month == 7]
    july_df[cluster] = month_data.groupby(month_data.index.hour).mean()
july_df.to_csv('../data/loadshapes/elmas_data_july_24hr_profile.csv')