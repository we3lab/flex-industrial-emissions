import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import zipfile
import gdown
import shutil
from pathlib import Path

# Load CSV with file names and download links
LBNL_BASE_DIR = Path('data/loadshapes/lbnl/phase4_data_release')
LBNL_BASE_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv('data/loadshapes/lbnl_zip_files.csv')


def aggregate_cluster_data(csv_file, year, daily_averages_dir):
    """
    Load a cluster CSV file, aggregate to daily averages for January and July,
    and save the aggregated data. Returns the aggregated dataframes.
    """
    cluster_name = csv_file.stem
    january_file = daily_averages_dir / f'{cluster_name}_january_24hr.csv'
    july_file = daily_averages_dir / f'{cluster_name}_july_24hr.csv'
    
    # Check if already processed
    if january_file.exists() and july_file.exists():
        january_df = pd.read_csv(january_file, index_col=0)
        july_df = pd.read_csv(july_file, index_col=0)
        return january_df, july_df
    
    # Load cluster data
    df_cluster = pd.read_csv(csv_file)
    n_hours = len(df_cluster)
    
    # Create datetime index based on actual data length
    start_date = pd.Timestamp(f'{year}-01-01 00:00:00')
    datetime_index = pd.date_range(start=start_date, periods=n_hours, freq='h')
    cluster_data = pd.Series(df_cluster['total'].values, index=datetime_index)
    
    # Aggregate January data
    january_data = cluster_data[cluster_data.index.month == 1]
    january_weekday = january_data[january_data.index.dayofweek < 5]
    january_grouped = january_weekday.groupby(january_weekday.index.hour)
    january_df = pd.DataFrame({
        'mean': january_grouped.mean(),
        'std': january_grouped.std()
    })
    january_df.to_csv(january_file)
    
    # Aggregate July data
    july_data = cluster_data[cluster_data.index.month == 7]
    july_weekday = july_data[july_data.index.dayofweek < 5]
    july_grouped = july_weekday.groupby(july_weekday.index.hour)
    july_df = pd.DataFrame({
        'mean': july_grouped.mean(),
        'std': july_grouped.std()
    })
    july_df.to_csv(july_file)
    
    return january_df, july_df


def process_time_period(data, time_period, day_period):
    """
    Retrieve the subset of the data for the given month 
    and return the mean load for each hour.
    """
    period_data = data.loc[data.index.month == (1 if time_period == 'January' else 7)]
    weekday_data = period_data.loc[period_data.index.dayofweek < day_period]
    grouped = weekday_data.groupby(weekday_data.index.hour)
    return pd.DataFrame({'mean': grouped.mean(), 'std': grouped.std()})

def plot_load_shapes(aggregated_data, cluster_names, plot_type, day_period=5, num_clusters = 18, output_dir='data/loadshapes/lbnl', scenario_name=''):
    """
    Plot the load shapes for the LBNL data using daily averages.
    aggregated_data: list of (january_df, july_df) tuples
    """
    nrows = (num_clusters + 2) // 3
    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 6 * nrows))
    axs = axs.flatten()
    
    max_load = 0
    for january_df, july_df in aggregated_data:
        max_load = max(max_load, january_df['mean'].max(), july_df['mean'].max())

    for i, (january_df, july_df) in enumerate(aggregated_data):
        if i >= num_clusters:
            break
        
        if plot_type == 'normalized':
            january_mean = january_df['mean'] / january_df['mean'].mean()
            january_std = january_df['std'] / january_df['mean'].mean()
            july_mean = july_df['mean'] / july_df['mean'].mean()
            july_std = july_df['std'] / july_df['mean'].mean()
            ylim = (0.0, 2.5)
        else:
            january_mean = january_df['mean']
            january_std = january_df['std']
            july_mean = july_df['mean']
            july_std = july_df['std']
            ylim = (0, max_load * 1.1)
        
        axs[i].plot(january_mean.index, january_mean, label='January', color='blue')
        axs[i].fill_between(january_mean.index, january_mean - january_std, january_mean + january_std, alpha=0.2, color='blue')
        axs[i].plot(july_mean.index, july_mean, label='July', color='red')
        axs[i].fill_between(july_mean.index, july_mean - july_std, july_mean + july_std, alpha=0.2, color='red')
        
        cluster_name = cluster_names[i] if i < len(cluster_names) else f'Cluster {i + 1}'
        axs[i].set_title(f'{i + 1}: {cluster_name}', fontsize=10)
        axs[i].set_xlabel('Hour')
        axs[i].set_ylabel('Load')
        axs[i].set_ylim(ylim)
        axs[i].legend()
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(12))

    title = f'{"Normalized" if plot_type == "normalized" else "Original"} Load Shapes: January vs July'
    if scenario_name:
        title = f'{scenario_name} - {title}'
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f'{scenario_name}_{plot_type}.png' if scenario_name else Path(output_dir) / f'{plot_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# # Check each file
# for _, row in df.iterrows():
#     file_name = row['file_name']
#     download_link = row['download_link']
    
#     # Check if folder already exists (unzipped)
#     folder_name = file_name.replace('.zip', '')
#     folder_path = LBNL_BASE_DIR / folder_name
    
#     if folder_path.exists() and any(folder_path.iterdir()):
#         print(f"{folder_name} already exists")
#     else:
#         print(f"downloading {folder_name}")
#         zip_path = extract_dir / file_name
#         extract_to = extract_dir / file_name.replace('.zip', '')
                
#         if 'uc?id=' in download_link:
#             print("direct file download")
#             # Direct file download link with file ID
#             gdown.download(download_link, str(zip_path), quiet=False)
#         else:
#             print("folder link")
#             # Folder link - download specific file from folder
#             folder_url = f'https://drive.google.com/drive/folders/{folder_id}?usp=sharing'
#             temp_dir = extract_dir / 'temp_download'
#             temp_dir.mkdir(exist_ok=True)
#             gdown.download_folder(folder_url, output=str(temp_dir), quiet=False, use_cookies=False)
#             # Find the downloaded file
#             for f in temp_dir.rglob(file_name):
#                 f.rename(zip_path)
#                 break
#             # Clean up temp directory
#             for item in temp_dir.iterdir():
#                 if item != zip_path.parent:
#                     if item.is_file():
#                         item.unlink()
#                     elif item.is_dir():
#                         shutil.rmtree(item)
#             temp_dir.rmdir()
        
#         # Extract zip file and remove
#         print(f"Extracting {file_name}")
#         extract_to.mkdir(exist_ok=True)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_to)
#         zip_path.unlink() # delete
#         print(f"Extracted {file_name}")


# Plotting all LBNL scenarios
print("Plotting LBNL loadshapes")
for scenario_folder in LBNL_BASE_DIR.iterdir():
    if scenario_folder.is_dir() and scenario_folder.name.startswith('anonymized'):
        scenario_name = scenario_folder.name
        
        # Extract year from scenario name (e.g., "2050" from "anonymized-1in2-midDemand-with_EE_FS-2050")
        year_match = [int(s) for s in scenario_name.split('-') if s.isdigit() and len(s) == 4]
        year = year_match[0] if year_match else 2025
        cluster_summary = pd.read_csv(scenario_folder / 'cluster_summary.csv')
        
        # Print all unique sectors
        unique_sectors = sorted(cluster_summary['sector'].unique())
        print(f"sectors in {scenario_name}:")
        for sector in unique_sectors:
            print(f"  - {sector}")
        
        
        
        # Get all CSV files (clusters)
        all_csv_files = sorted([f for f in scenario_folder.glob('*.csv') if f.name != 'cluster_summary.csv'])
        
        # Create mapping from file name to summary row
        summary_dict = {row['name']: row for _, row in cluster_summary.iterrows()}
        
        # Filter csv_files to only include target types
        csv_files = []
        target_types = ['food_bev', 'materials', 'datacenter']
        for csv_file in all_csv_files:
            cluster_name = csv_file.stem
            if cluster_name in summary_dict:
                row = summary_dict[cluster_name]
                building_type = str(row['building_type']).lower()
                if any(target_type.lower() in building_type for target_type in target_types):
                    csv_files.append(csv_file)
        
        # Print other industrial sector building types
        industrial_types = []
        for csv_file in all_csv_files:
            cluster_name = csv_file.stem
            if cluster_name in summary_dict:
                row = summary_dict[cluster_name]
                if row['sector'] == 'ind':
                    building_type = str(row['building_type'])
                    building_type_lower = building_type.lower()
                    if not any(target_type.lower() in building_type_lower for target_type in target_types):
                        industrial_types.append(building_type)
        
        industrial_types_unique = sorted(set(industrial_types))
        print(f"industrial sector building types in {scenario_name}:")
        for bt in industrial_types_unique:
            print(f"  - {bt}")
        
        # Create daily_averages subfolder
        daily_averages_dir = scenario_folder / 'daily_averages'
        daily_averages_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {scenario_name}: {len(csv_files)} clusters (filtered: {', '.join(target_types)})")
        
        # Process each cluster individually and aggregate
        processed_count = 0
        skipped_count = 0
        
        # Group clusters by target type
        clusters_by_type = {target_type: [] for target_type in target_types}
        
        for csv_file in csv_files:
            cluster_name = csv_file.stem
            
            # Find which target_type this cluster matches
            matched_type = None
            if cluster_name in summary_dict:
                row = summary_dict[cluster_name]
                building_type = str(row['building_type']).lower()
                for target_type in target_types:
                    if target_type.lower() in building_type:
                        matched_type = target_type
                        break
            
            if matched_type:
                clusters_by_type[matched_type].append((csv_file, cluster_name))
        
        # Create output directory
        output_dir = Path('data/loadshapes/lbnl') / scenario_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and plot each target type separately
        for target_type in target_types:
            type_clusters = clusters_by_type[target_type]
            
            print(f"\nProcessing {target_type}: {len(type_clusters)} clusters")
            aggregated_data = []
            cluster_file_names = []
            
            for csv_file, cluster_name in type_clusters:
                cluster_file_names.append(cluster_name)
                
                # Check if already processed
                january_file = daily_averages_dir / f'{cluster_name}_january_24hr.csv'
                july_file = daily_averages_dir / f'{cluster_name}_july_24hr.csv'
                if january_file.exists() and july_file.exists():
                    skipped_count += 1
                    january_df = pd.read_csv(january_file, index_col=0)
                    july_df = pd.read_csv(july_file, index_col=0)
                    aggregated_data.append((january_df, july_df))
                else:
                    processed_count += 1
                    january_df, july_df = aggregate_cluster_data(csv_file, year, daily_averages_dir)
                    aggregated_data.append((january_df, july_df))
            
            # Generate cluster names from summary, matching file names
            name_columns = ['sector', 'util', 'building_type', 'size', 'climate', 'care', 'lca', 'lshp', 'kwh_bin']
            cluster_names = []
            
            for cluster_file_name in cluster_file_names:
                row = summary_dict[cluster_file_name]
                name_parts = []
                for col in name_columns:
                    name_parts.append(str(row[col]))
                cluster_name = '-'.join(name_parts)
                cluster_names.append(cluster_name)
            
            # Plot original and normalized for this target type
            plot_scenario_name = f'{scenario_name}_{target_type}'
            print(f"Plotting {target_type}")
            plot_load_shapes(aggregated_data, cluster_names, 'original', day_period=5, num_clusters=len(aggregated_data), 
                             output_dir=output_dir, scenario_name=plot_scenario_name)
            plot_load_shapes(aggregated_data, cluster_names, 'normalized', day_period=5, num_clusters=len(aggregated_data), 
                             output_dir=output_dir, scenario_name=plot_scenario_name)