import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

total_precipitation_per_city = totals.sum(1) 
lowest_precipitation_city = total_precipitation_per_city.idxmin()  
print("City with lowest total precipitation:")
print(lowest_precipitation_city)

total_precipitation_per_month = totals.sum(0) 
total_observations_per_month = counts.sum(0)  
average_precipitation_per_month = total_precipitation_per_month / total_observations_per_month
print("Average precipitation in each month:")
print(average_precipitation_per_month)

average_precipitation_per_city = total_precipitation_per_city / counts.sum(1)
print("Average precipitation in each city:")
print(average_precipitation_per_city)

