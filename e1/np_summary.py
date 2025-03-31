import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

total_precipitation_per_city = totals.sum(axis=1) 
lowest_precipitation_city = np.argmin(total_precipitation_per_city) 
print("Row with lowest total precipitation:")
print(lowest_precipitation_city)

total_precipitation_per_month = totals.sum(axis=0) 
total_observations_per_month = counts.sum(axis=0) 
average_precipitation_per_month = total_precipitation_per_month / total_observations_per_month
print("Average precipitation in each month:")
print(average_precipitation_per_month)

average_precipitation_per_city = total_precipitation_per_city / counts.sum(axis=1)
print("Average precipitation in each city:")
print(average_precipitation_per_city)

quarters = totals.reshape(totals.shape[0], 4, 3).sum(axis=2)  
print("Quarterly precipitation totals:")
print(quarters)