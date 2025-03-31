import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def distance(city, stations):
    lat1, lon1 = np.radians(city['latitude']), np.radians(city['longitude'])
    lat2, lon2 = np.radians(stations['latitude']), np.radians(stations['longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c


def best_tmax(city, stations):
    distances = distance(city, stations)
    best_index = distances.idxmin()
    return stations.loc[best_index, 'avg_tmax'] / 10


def main(stations_file, city_file, output_file):
    stations = pd.read_json(stations_file, lines=True)
    cities = pd.read_csv(city_file)

    cities = cities.dropna(subset=['population', 'area'])
    cities['area_km2'] = cities['area'] / 1e6
    cities = cities[cities['area_km2'] <= 10000]

    cities['density'] = cities['population'] / cities['area_km2']
    cities['avg_tmax'] = cities.apply(best_tmax, stations=stations, axis=1)

    plt.scatter(cities['avg_tmax'], cities['density'])
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.title('Temperature vs Population Density')
    plt.savefig(output_file)

if __name__ == '__main__':
    stations_file = sys.argv[1]
    city_file = sys.argv[2]
    output_file = sys.argv[3]
    main(stations_file, city_file, output_file)