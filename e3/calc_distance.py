import sys
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from math import radians, cos, sin, sqrt, asin
from xml.dom.minidom import parse

def read_gpx(filename):
    dom = parse(filename)
    points = []

    for trkpt in dom.getElementsByTagName('trkpt'):
        lat = float(trkpt.getAttribute('lat'))
        lon = float(trkpt.getAttribute('lon'))
        points.append({'lat': lat, 'lon': lon})

    return pd.DataFrame(points)

"""
https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
Haversine implementation referenced from webpage
"""
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return r * c

def distance(points):
    temp_points = points.copy()

    temp_points['lat2'] = temp_points['lat'].shift(-1)
    temp_points['lon2'] = temp_points['lon'].shift(-1)

    distances = temp_points.dropna(subset=['lat2', 'lon2']).apply(
        lambda row: haversine(row['lat'], row['lon'], row['lat2'], row['lon2']),
        axis=1
    )
    return distances.sum()

def smooth(points):
    initial_state = points.iloc[0]
    observation_covariance = np.diag([17.5/100000, 17.5/100000]) ** 2
    transition_covariance = np.diag([10/100000, 10/100000]) ** 2
    transition = [[1, 0], [0, 1]]

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    smoothed_state_means, _ = kf.smooth(points[['lat', 'lon']].values)
    return pd.DataFrame(smoothed_state_means, columns=['lat', 'lon'])


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def main():
    points = read_gpx(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))

    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')

if __name__ == '__main__':
    main()
