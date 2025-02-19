import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import folium
from Apriori import apriori
from folium.plugins import DualMap, HeatMap


# ============================== Part 1: Data Preprocessing ============================== #


# Read in the data
def read_data(filename):
    """
    Read in the data from a csv file
    :param filename: The name of the file to read
    :return: Dataframe of the data
    """
    crash_data = pd.read_csv(filename, low_memory=False)
    crash_data['CRASH DATE'] = pd.to_datetime(crash_data['CRASH DATE'], format='%m/%d/%Y')
    crash_data['CRASH TIME'] = pd.to_datetime(crash_data['CRASH TIME'], format='%H:%M')
    return crash_data


def clean_data_based_on_year(crash_data):
    """
    Clean the data based on the just two years: 2019 and 2020.
    :param crash_data: NYC crash data in the form of a dataframe
    :return:
    """
    year_filtered_data = crash_data.loc[
        (crash_data['CRASH DATE'].dt.year == 2019) | (crash_data['CRASH DATE'].dt.year == 2020)]
    return year_filtered_data


def clean_data_based_on_date(crash_data):
    """
    Clean the data based on the date. June and July, of just two years: 2019 and 2020.
    :param crash_data: NYC crash data in the form of a dataframe
    :return:
    """
    date_filtered_data = crash_data.loc[
        (crash_data['CRASH DATE'].dt.month == 6) | (crash_data['CRASH DATE'].dt.month == 7)]
    return date_filtered_data


def brooklyn_data(filtered_crash_data):
    """
    Filter the data to only include crashes that happened in Brooklyn
    :param filtered_crash_data: Filtered data based on date
    :return: Dataframe of the filtered data
    """

    brooklyn_crash_data = filtered_crash_data[filtered_crash_data['BOROUGH'] == 'BROOKLYN']
    brooklyn_crash_data = brooklyn_crash_data.drop(['BOROUGH'], axis=1)
    return brooklyn_crash_data


def cleanup_data(unfiltered_brooklyn_crash_data):
    """
    Clean up the data to only include the columns we are interested in
    :param unfiltered_brooklyn_crash_data: Filtered data only based on date and borough
    :return: Dataframe of the cleaned up data
    """
    brooklyn_crash_data = unfiltered_brooklyn_crash_data.copy()

    # Add a column for total killed and total injured
    total_killed = brooklyn_crash_data.columns[brooklyn_crash_data.columns.str.contains('KILLED')]
    total_injured = brooklyn_crash_data.columns[brooklyn_crash_data.columns.str.contains('INJURED')]
    brooklyn_crash_data["TOTAL KILLED"] = brooklyn_crash_data[total_killed].sum(axis=1).astype(int)
    brooklyn_crash_data["TOTAL INJURED"] = brooklyn_crash_data[total_injured].sum(axis=1).astype(int)

    # Add a column for contributing factors
    contributing_factors = brooklyn_crash_data.columns[brooklyn_crash_data.columns.str.contains('CONTRIBUTING FACTOR')]
    brooklyn_crash_data.loc[:, 'CONTRIBUTING FACTORS'] = brooklyn_crash_data[contributing_factors].apply(
        lambda row: [value for value in row if not pd.isna(value) and value != "Unspecified"],
        axis=1)

    # Add a column for vehicle types
    vehicle_types = brooklyn_crash_data.columns[brooklyn_crash_data.columns.str.contains('VEHICLE TYPE')]
    brooklyn_crash_data['VEHICLE TYPES'] = brooklyn_crash_data[vehicle_types].apply(lambda row: row.dropna().tolist(),
                                                                                    axis=1)

    columns_to_drop = list(total_killed) + list(total_injured) + list(contributing_factors) + list(vehicle_types) + [
        "LOCATION", "ZIP CODE","COLLISION_ID"]
    brooklyn_crash_data = brooklyn_crash_data.drop(columns_to_drop, axis=1)

    return brooklyn_crash_data


# ============================================================================================== #


# ============================== Part 2: Data Analysis ========================================= #

def draw_map(freq_data_2019, freq_data_2020, severity_data_2019, severity_data_2020):
    nyc_map_freq = DualMap(
        location=[40.650002, -73.949997],
        zoom_start=12,
        tiles='https://tile.jawg.io/jawg-dark/{z}/{x}/{y}{r}.png?access-token=C4fJsGdUYZJD2f8mmCocQQpaYf0yG6BLuKZHdKgIQIo3hCdeiHRJTDVU4xv9npah',
        attr='<a href="https://www.jawg.io/">Jawg</a>'
    )

    nyc_map_severity = DualMap(
        location=[40.650002, -73.949997],
        zoom_start=12,
        tiles='https://tile.jawg.io/jawg-dark/{z}/{x}/{y}{r}.png?access-token=C4fJsGdUYZJD2f8mmCocQQpaYf0yG6BLuKZHdKgIQIo3hCdeiHRJTDVU4xv9npah',
        attr='<a href="https://www.jawg.io/">Jawg</a>'
    )

    for index, row in freq_data_2019.iterrows():
        if row['ACCIDENT_COUNT'] > 1:
            # Calculate the radius based on the number of accidents at that location
            radius = row['ACCIDENT_COUNT'] * 2  # Adjust this multiplier for better visualization

            # Create a Circle Marker
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=radius,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Location:({row['LATITUDE'], row['LONGITUDE']}), Accident Count: {row['ACCIDENT_COUNT']}"
            ).add_to(nyc_map_freq.m1)

    for index, row in freq_data_2020.iterrows():
        if row['ACCIDENT_COUNT'] > 1:
            # Calculate the radius based on the number of accidents at that location
            radius = row['ACCIDENT_COUNT'] * 2  # Adjust this multiplier for better visualization

            # Create a Circle Marker
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=radius,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f"Location:({row['LATITUDE'], row['LONGITUDE']}), Accident Count: {row['ACCIDENT_COUNT']}"
            ).add_to(nyc_map_freq.m2)

    for index, row in severity_data_2019.iterrows():
        # Calculate the radius based on the number of accidents at that location
        if row['TOTAL KILLED'] + row['TOTAL INJURED'] > 2:
            radius = (row['TOTAL KILLED'] + row['TOTAL INJURED']) * 2  # Adjust this multiplier for better visualization

            # Create a Circle Marker
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=radius,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=f"Location:({row['LATITUDE'], row['LONGITUDE']}), Total Killed: {row['TOTAL KILLED']}, Total Injured: {row['TOTAL INJURED']}"
            ).add_to(nyc_map_severity.m1)

    for index, row in severity_data_2020.iterrows():
        # Calculate the radius based on the number of accidents at that location
        if row['TOTAL KILLED'] + row['TOTAL INJURED'] > 2:
            radius = (row['TOTAL KILLED'] + row['TOTAL INJURED']) * 2  # Adjust this multiplier for better visualization

            # Create a Circle Marker
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=radius,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f"Location:({row['LATITUDE'], row['LONGITUDE']}), (Total Killed: {row['TOTAL KILLED']}, Total Injured: {row['TOTAL INJURED']}"
            ).add_to(nyc_map_severity.m2)

    return nyc_map_freq, nyc_map_severity


def heatmap(data):
    """
    Draw a heatmap of the given data
    :param data: The data to draw the heatmap for
    :param title: The title of the heatmap
    :return: The heatmap
    """
    # Create a map
    nyc_map = folium.Map(
        location=[40.650002, -73.949997],
        zoom_start=12,
        tiles='https://tile.jawg.io/jawg-dark/{z}/{x}/{y}{r}.png?access-token=C4fJsGdUYZJD2f8mmCocQQpaYf0yG6BLuKZHdKgIQIo3hCdeiHRJTDVU4xv9npah',
        attr='<a href="https://www.jawg.io/">Jawg</a>'
    )

    # Create a heatmap layer
    HeatMap(data=data, radius=15).add_to(nyc_map)

    # Display the map
    return nyc_map


def filter_data_based_on_subset(df, subset):
    """
    Filter the data based on the given subset
    :param data: The data to filter
    :param subset: The subset to filter the data on
    :return: The filtered data
    """

    # Filter rows based on the condition
    filtered_rows = df[df['VEHICLE TYPES'].apply(lambda x: set(subset).issubset(x))]
    return filtered_rows


# Question 1
def difference_in_summer_2019_2020(brooklyn_data):
    """
    Difference in crashes between Summer 2019 and Summer 2020
    :param brooklyn_data: Dataframe containing the crash data for Brooklyn
    :return:
    """
    # Filter data for June 2019
    summer_2019_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2019-06-01') & (brooklyn_data['CRASH DATE'] <= '2019-08-31')]
    summer_2019_data = summer_2019_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Filter data for June 2020
    summer_2020_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2020-06-01') & (brooklyn_data['CRASH DATE'] <= '2020-08-31')]
    summer_2020_data = summer_2020_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    location_counts_2019 = summer_2019_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')
    location_counts_2020 = summer_2020_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')

    # Apply Apriori algorithm to find the association rules for Summer 2019 and Summer 2020
    min_support = 50  # Number of accidents to be considered frequent
    min_confidence = 0.5  # Minimum confidence for a rule to be considered strong

    vehicle_type_subset_counts_2019, vehicle_type_association_rules_2019 = apriori(
        summer_2019_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    contributing_factors_2019, contributing_factors_association_rules_2019 = apriori(
        summer_2019_data['CONTRIBUTING FACTORS'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2019 = {key: value for key, value in vehicle_type_subset_counts_2019.items() if
                                       len(key) > 1}
    top_5_vehicle_type_subset_counts_2019 = sorted(vehicle_type_subset_counts_2019.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]
    contributing_factors_2019 = {key: value for key, value in contributing_factors_2019.items() if len(key) > 1}
    top_contributing_factor_2019 = (sorted(contributing_factors_2019.items(), key=lambda x: x[1], reverse=True)[0])
    most_common_vehicle_type_accident_2019 = top_5_vehicle_type_subset_counts_2019[0][0]
    filtered_data_2019 = filter_data_based_on_subset(summer_2019_data, most_common_vehicle_type_accident_2019)
    heatmap(filtered_data_2019[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JUNE2019.html')

    vehicle_type_subset_counts_2020, vehicle_type_association_rules_2020 = apriori(
        summer_2020_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    contributing_factors_2020, contributing_factors_association_rules_2020 = apriori(
        summer_2020_data['CONTRIBUTING FACTORS'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2020 = {key: value for key, value in vehicle_type_subset_counts_2020.items() if
                                       len(key) > 1}
    top_contributing_factor_2020 = (sorted(contributing_factors_2020.items(), key=lambda x: x[1], reverse=True)[0])
    top_5_vehicle_type_subset_counts_2020 = sorted(vehicle_type_subset_counts_2020.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]

    most_common_vehicle_type_accident_2020 = top_5_vehicle_type_subset_counts_2020[0][0]
    filtered_data_2020 = filter_data_based_on_subset(summer_2020_data, most_common_vehicle_type_accident_2020)
    heatmap(filtered_data_2020[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JUNE2020.html')

    print("Summer 2019")
    print("---------")
    print(f"Total number of accidents:{location_counts_2019['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2019['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {summer_2019_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {summer_2019_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2019:
        print(f"\tVehicle Types: {subset}, Count: {count}")
    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2019:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")
    print(f"Most common contributing factor: {top_contributing_factor_2019[0]}")
    print("---------")

    print("Summer 2020")
    print("---------")
    print(f"Total number of accidents:{location_counts_2020['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2020['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {summer_2020_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {summer_2020_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2020:
        print(f"\tVehicle Types: {subset}, Count: {count}")
    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2020:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")
    print(f"Most common contributing factor: {top_contributing_factor_2020[0]}")

    nyc_map_freq, nyc_map_severity = draw_map(location_counts_2019, location_counts_2020, summer_2019_data,
                                              summer_2020_data)

    nyc_map_freq.save('brooklyn_accidents_frequency_SUMMER.html')
    nyc_map_severity.save('brooklyn_accidents_severity_SUMMER.html')


# Question 2
def difference_in_June_2019_2020(brooklyn_data):
    """
    Difference in crashes between June 2019 and June 2020
    :param brooklyn_data:
    :return:
    """

    # Filter data for June 2019
    june_2019_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2019-06-01') & (brooklyn_data['CRASH DATE'] <= '2019-06-30')]
    june_2019_data = june_2019_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Filter data for June 2020
    june_2020_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2020-06-01') & (brooklyn_data['CRASH DATE'] <= '2020-06-30')]
    june_2020_data = june_2020_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    location_counts_2019 = june_2019_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')
    location_counts_2020 = june_2020_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')

    # Apply Apriori algorithm to find the association rules for June 2019 and June 2020
    min_support = 20  # Number of accidents to be considered frequent
    min_confidence = 0.4  # Minimum confidence for a rule to be considered strong

    vehicle_type_subset_counts_2019, vehicle_type_association_rules_2019 = apriori(
        june_2019_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2019 = {key: value for key, value in vehicle_type_subset_counts_2019.items() if
                                       len(key) > 1}
    top_5_vehicle_type_subset_counts_2019 = sorted(vehicle_type_subset_counts_2019.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]
    most_common_vehicle_type_accident_2019 = top_5_vehicle_type_subset_counts_2019[0][0]
    filtered_data_2019 = filter_data_based_on_subset(june_2019_data, most_common_vehicle_type_accident_2019)
    heatmap(filtered_data_2019[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JUNE2019.html')

    vehicle_type_subset_counts_2020, vehicle_type_association_rules_2020 = apriori(
        june_2020_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2020 = {key: value for key, value in vehicle_type_subset_counts_2020.items() if
                                       len(key) > 1}
    top_5_vehicle_type_subset_counts_2020 = sorted(vehicle_type_subset_counts_2020.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]

    most_common_vehicle_type_accident_2020 = top_5_vehicle_type_subset_counts_2020[0][0]
    filtered_data_2020 = filter_data_based_on_subset(june_2020_data, most_common_vehicle_type_accident_2020)
    heatmap(filtered_data_2020[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JUNE2020.html')

    print("June 2019")
    print("---------")
    print(f"Total number of accidents:{location_counts_2019['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2019['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {june_2019_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {june_2019_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2019:
        print(f"\tVehicle Types: {subset}, Count: {count}")
    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2019:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")
    print("---------")

    print("June 2020")
    print("---------")
    print(f"Total number of accidents:{location_counts_2020['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2020['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {june_2020_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {june_2020_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2020:
        print(f"\tVehicle Types: {subset}, Count: {count}")

    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2020:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")

    nyc_map_freq, nyc_map_severity = draw_map(location_counts_2019, location_counts_2020, june_2019_data,
                                              june_2020_data)

    nyc_map_freq.save('brooklyn_accidents_frequency_JUNE.html')
    nyc_map_severity.save('brooklyn_accidents_severity_JUNE.html')


def difference_in_July_2019_2020(brooklyn_data):
    """
    Difference in crashes between July 2019 and July 2020
    :param brooklyn_data:
    :return:
    """

    # Filter data for July 2019
    july_2019_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2019-07-01') & (brooklyn_data['CRASH DATE'] <= '2019-07-31')]
    july_2019_data = july_2019_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Filter data for July 2020
    july_2020_data = brooklyn_data[
        (brooklyn_data['CRASH DATE'] >= '2020-07-01') & (brooklyn_data['CRASH DATE'] <= '2020-07-31')]
    july_2020_data = july_2020_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    location_counts_2019 = july_2019_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')
    location_counts_2020 = july_2020_data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')

    # Apply Apriori algorithm to find the association rules for June 2019 and June 2020
    min_support = 20  # Number of accidents to be considered frequent
    min_confidence = 0.4  # Minimum confidence for a rule to be considered strong

    vehicle_type_subset_counts_2019, vehicle_type_association_rules_2019 = apriori(
        july_2019_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2019 = {key: value for key, value in vehicle_type_subset_counts_2019.items() if
                                       len(key) > 1}
    top_5_vehicle_type_subset_counts_2019 = sorted(vehicle_type_subset_counts_2019.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]
    most_common_vehicle_type_accident_2019 = top_5_vehicle_type_subset_counts_2019[0][0]
    filtered_data_2019 = filter_data_based_on_subset(july_2019_data, most_common_vehicle_type_accident_2019)
    heatmap(filtered_data_2019[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JULY2019.html')

    vehicle_type_subset_counts_2020, vehicle_type_association_rules_2020 = apriori(
        july_2020_data['VEHICLE TYPES'].tolist(),
        min_support, min_confidence)
    vehicle_type_subset_counts_2020 = {key: value for key, value in vehicle_type_subset_counts_2020.items() if
                                       len(key) > 1}
    top_5_vehicle_type_subset_counts_2020 = sorted(vehicle_type_subset_counts_2020.items(), key=lambda x: x[1],
                                                   reverse=True)[:5]
    most_common_vehicle_type_accident_2020 = top_5_vehicle_type_subset_counts_2020[0][0]
    filtered_data_2020 = filter_data_based_on_subset(july_2020_data, most_common_vehicle_type_accident_2020)
    heatmap(filtered_data_2020[['LATITUDE', 'LONGITUDE']].values.tolist()).save('most_common_accident_JULY2020.html')

    print("July 2019")
    print("---------")
    print(f"Total number of accidents:{location_counts_2019['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2019['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2019.loc[location_counts_2019['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {july_2019_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {july_2019_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2019:
        print(f"\tVehicle Types: {subset}, Count: {count}")
    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2019:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")
    print("---------")

    print("July 2020")
    print("---------")
    print(f"Total number of accidents:{location_counts_2020['ACCIDENT_COUNT'].sum()}")
    print(f"Max number of accidents: {location_counts_2020['ACCIDENT_COUNT'].max()}")
    print(
        f"Location with max accidents: ({location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LATITUDE']}, "
        f"{location_counts_2020.loc[location_counts_2020['ACCIDENT_COUNT'].idxmax()]['LONGITUDE']})")
    print(f"Number of persons killed: {july_2020_data['TOTAL KILLED'].sum()}")
    print(f"Number of persons injured: {july_2020_data['TOTAL INJURED'].sum()}")
    print("Most frequent types of accident:-")
    for subset, count in top_5_vehicle_type_subset_counts_2020:
        print(f"\tVehicle Types: {subset}, Count: {count}")
    print("Most probable types of accidents:-")
    for rule, confidence in vehicle_type_association_rules_2020:
        print(f"\t{rule[0]} -> {rule[1]}, accident probability (confidence): {confidence}")

    nyc_map_freq, nyc_map_severity = draw_map(location_counts_2019, location_counts_2020, july_2019_data,
                                              july_2020_data)

    nyc_map_freq.save('brooklyn_accidents_frequency_JULY.html')
    nyc_map_severity.save('brooklyn_accidents_severity_JULY.html')


def worse_100_consecutive_days(brooklyn_data):
    """
    Consecutive 100 days with the most accidents.
    :param brooklyn_data: Dataframe containing the crash data for Brooklyn
    :return:
    """

    jan19_oct20_data = brooklyn_data.loc[
        (brooklyn_data['CRASH DATE'] >= '2019-01-01') & (brooklyn_data['CRASH DATE'] <= '2020-10-31')].copy()

    jan19_oct20_data.sort_values(['CRASH DATE', 'CRASH TIME'], inplace=True)

    # Create overlapping windows of 100 consecutive days
    window_size = 100
    windows = [(start, start + pd.Timedelta(days=window_size - 1)) for start in
               pd.date_range(start=jan19_oct20_data['CRASH DATE'].min(), end=jan19_oct20_data['CRASH DATE'].max(), freq=pd.DateOffset(days=1))[
               :-window_size]]

    # Count accidents for each window
    accidents_per_window = []

    for window_start, window_end in windows:
        window_data = jan19_oct20_data[(jan19_oct20_data['CRASH DATE'] >= window_start) & (jan19_oct20_data['CRASH DATE'] <= window_end)]
        accidents_count = len(window_data)
        accidents_per_window.append(
            {'WINDOW_START': window_start, 'WINDOW_END': window_end, 'ACCIDENT_COUNT': accidents_count})

    # Create a DataFrame for the results
    result_df = pd.DataFrame(accidents_per_window)

    top_100_days = result_df.sort_values('ACCIDENT_COUNT', ascending=False).head(1)

    # Plot the result
    grouped_data = jan19_oct20_data.groupby('CRASH DATE').size().reset_index(name='ACCIDENT COUNT')

    # Highlight the top 100 consecutive days
    top_100_start = top_100_days['WINDOW_START'].iloc[0].date()
    top_100_end = top_100_days['WINDOW_END'].iloc[0].date()

    # Plot the result
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    ax.xaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    plt.plot(grouped_data['CRASH DATE'], grouped_data['ACCIDENT COUNT'])
    plt.axvspan(top_100_start, top_100_end, color='red', alpha=0.3, label='Top 100 Days')
    plt.axvline(top_100_start, color='red', linestyle='--', label='Start date')
    plt.axvline(top_100_end, color='red', linestyle='--', label='End date')

    # Set the modified tick labels
    plt.xticks(rotation=45)
    plt.text(top_100_start, -15,top_100_start , color='red', rotation=45, ha='center')
    plt.text(top_100_end, -15, top_100_end, color='red', rotation=45, ha='center')
    plt.title('100 Consecutive Days with the Most Accidents')
    plt.tight_layout()
    plt.legend()
    plt.show()

    worse_days = f"{top_100_start} to {top_100_end} with {top_100_days['ACCIDENT_COUNT'].iloc[0]} accidents"

    return worse_days



def day_of_week_with_most_accidents(data):
    # Create a new column DAY_OF_WEEK with the day of the week for each date
    data['DAY_OF_WEEK'] = data['CRASH DATE'].dt.dayofweek

    accidents_by_day = data.groupby('DAY_OF_WEEK').size().reset_index(name='ACCIDENT COUNT')

    # Group the data by DAY_OF_WEEK
    grouped_data = data.groupby('DAY_OF_WEEK').agg({'TOTAL KILLED': 'sum', 'TOTAL INJURED': 'sum'}).reset_index()

    # Find the day of the week with the maximum number of accidents
    num_day_of_week_with_most_accidents = grouped_data.loc[grouped_data['TOTAL KILLED'].idxmax()]['DAY_OF_WEEK']

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    accidents_by_day['DAY_OF_WEEK'] = accidents_by_day['DAY_OF_WEEK'].apply(lambda x: days_of_week[x])
    day_with_max_accidents = accidents_by_day.loc[(accidents_by_day['ACCIDENT COUNT'].idxmax() , 'DAY_OF_WEEK')]
    # print(accidents_by_day)

    # Convert the day of the week number to a name
    day_of_week_with_most_accidents = days_of_week[num_day_of_week_with_most_accidents]

    # Plot the bar graph for number of accidents by day of the week
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='solid', alpha=0.2)

    plt.bar(accidents_by_day['DAY_OF_WEEK'], accidents_by_day['ACCIDENT COUNT'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Day of the Week')
    plt.tight_layout()
    plt.show()


    # Plot the bar graph for killed and injured by day of the week
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Total Killed
    ax[0].set_axisbelow(True)
    ax[0].yaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    ax[0].xaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    ax[0].bar(grouped_data['DAY_OF_WEEK'].map({i: day for i, day in enumerate(days_of_week)}),
              grouped_data['TOTAL KILLED'], color='red', alpha=0.9, label='Total Killed')
    ax[0].set_xlabel('Day of the Week')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Total Killed')
    ax[0].tick_params(axis='x', labelsize=8)

    # Plot Total Injured
    ax[1].set_axisbelow(True)
    ax[1].yaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    ax[1].xaxis.grid(color='gray', linestyle='solid', alpha=0.2)
    ax[1].bar(grouped_data['DAY_OF_WEEK'].map({i: day for i, day in enumerate(days_of_week)}),
              grouped_data['TOTAL INJURED'], color='blue', alpha=0.9, label='Total Injured')
    ax[1].set_xlabel('Day of the Week')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Total Injured')
    ax[1].tick_params(axis='x', labelsize=8)
    fig.suptitle('Total Killed and Injured by Day of the Week', fontsize=20)
    plt.tight_layout()
    plt.show()

    return day_of_week_with_most_accidents, accidents_by_day, day_with_max_accidents


def hour_of_day_with_most_accidents(data):
    """
    Determines which hour of the day has the most accidents.
    :param data: Dataframe containing the crash data.
    :return: The hour of the day (0-23) with the most accidents.
    """

    # Extract the hour from the 'CRASH TIME'
    data['HOUR OF DAY'] = data['CRASH TIME'].dt.hour

    # Group by the hour of the day and count the number of accidents
    accidents_by_hour = data.groupby('HOUR OF DAY').size()

    # Find the hour with the most accidents
    max_accidents_hour = accidents_by_hour.idxmax()

    # Convert the hour to a string
    if max_accidents_hour < 12:
        max_accidents_hour = str(max_accidents_hour) + " AM"
    elif max_accidents_hour == 12:
        max_accidents_hour = str(max_accidents_hour) + " PM"
    else:
        max_accidents_hour = str(max_accidents_hour - 12) + " PM"

    # Plot a histogram or bar graph
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='--', alpha=0.3)
    plt.bar(accidents_by_hour.index, accidents_by_hour.values, color='blue', alpha=0.9, width=0.8)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Hour of the Day')
    plt.xticks(accidents_by_hour.index)
    plt.tight_layout()
    plt.show()

    return max_accidents_hour


def top_12_accident_days_2020(data):
    """
    Finds the 12 days in 2020 with the most accidents.
    :param data: Dataframe containing the crash data.
    :return: A dataframe with the top 12 days and the number of accidents.
    """
    # Filter the data for the year 2020
    data_2020 = data[data['CRASH DATE'].dt.year == 2020]

    # Group by the date and count the number of accidents
    accidents_by_day = data_2020.groupby('CRASH DATE').size().reset_index(name='ACCIDENT COUNT')

    # Sort the days by the number of accidents in descending order and get the top 12
    top_12_days = accidents_by_day.sort_values(by='ACCIDENT COUNT', ascending=False).head(12).reset_index(drop=True)

    return top_12_days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The name of the file to read")
    arguments = parser.parse_args()
    crash_data = read_data(arguments.filename)

    year_filtered_data = clean_data_based_on_year(crash_data)
    date_filtered_data = clean_data_based_on_date(year_filtered_data)

    brooklyn_crash_data_2019_2020 = brooklyn_data(year_filtered_data)
    cleaned_brooklyn_crash_data_2019_2020 = cleanup_data(brooklyn_crash_data_2019_2020)

    brooklyn_crash_data_june_july = brooklyn_data(date_filtered_data)
    cleaned_brooklyn_crash_data_june_july = cleanup_data(brooklyn_crash_data_june_july)
    cleaned_brooklyn_crash_data_june_july.to_csv("Brooklyn_crash_data_june1920-july1920.csv", index=False)

    # Question 1: What is the difference in crashes between 2019 and 2020?
    print("Question 1")
    print("---------------------------------------------------------")
    difference_in_summer_2019_2020(cleaned_brooklyn_crash_data_2019_2020)
    print("---------------------------------------------------------")

    # Question 2: What is the difference in crashes between July 2019 and July 2020?
    print()
    print()
    print("Question 2")
    print("---------------------------------------------------------")
    difference_in_June_2019_2020(cleaned_brooklyn_crash_data_june_july)
    print("---------------------------------------------------------")

    # Question 3: What is the difference in crashes between June 2019 and June 2020?
    print()
    print()
    print("Question 3")
    print("---------------------------------------------------------")
    difference_in_July_2019_2020(cleaned_brooklyn_crash_data_june_july)
    print("---------------------------------------------------------")

    # Question 4: For the year of January 2019 to October 2020, which 100 consecutive days had the most accidents?
    print()
    print()
    print("Question 4")
    print("Worse 100 consecutive days are:")
    print("---------------------------------------------------------")
    print(worse_100_consecutive_days(cleaned_brooklyn_crash_data_2019_2020))
    print("---------------------------------------------------------")

    # Question 5: Which day of the week has the most accidents?
    day_of_week_with_most_accident, accidents_by_day, day_with_max_accidents = day_of_week_with_most_accidents(
        cleaned_brooklyn_crash_data_2019_2020)
    print()
    print("Question 5")
    print("---------------------------------------------------------")
    print(f"Day of the Week with most accidents: {day_with_max_accidents}")
    print("---------------------------------------------------------")
    print(
        f"Day of the Week with most accidents conidering the number of people killed or injured: {day_of_week_with_most_accident}")
    print("---------------------------------------------------------")
    print(f"Week data: {accidents_by_day}")
    print("---------------------------------------------------------")

    # Question 6: Which hour of the day has the most accidents?
    hour_with_most_accidents = hour_of_day_with_most_accidents(cleaned_brooklyn_crash_data_2019_2020)
    print()
    print()
    print("Question 6")
    print(f"Hour of the day with most accidents: {hour_with_most_accidents}")
    print("---------------------------------------------------------")

    # Question 7: In the year 2020, which 12 days had the most accidents?
    print()
    print()
    print("Question 7")
    top_12_days_2020 = top_12_accident_days_2020(cleaned_brooklyn_crash_data_2019_2020)
    print("12 days that had the most accidents are: ")
    print()
    print(top_12_days_2020)
    print("---------------------------------------------------------")


if __name__ == "__main__":
    main()
