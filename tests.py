# Importing libraries

import numpy as np
import pandas as pd
import scipy as stats

# calculateMahalanobis function to calculate
# the Mahalanobis distance
def calculateMahalanobis(y=None, data=None, cov=None):

	y_mu = y - np.mean(data)
	if not cov:
		cov = np.cov(data.values.T)
	inv_covmat = np.linalg.inv(cov)
	left = np.dot(y_mu, inv_covmat)
	mahal = np.dot(left, y_mu.T)
	return mahal.diagonal()

# data
data = { 'Price': [100000, 800000, 650000, 700000,
				860000, 730000, 400000, 870000,
				780000, 400000],
		'Distance': [16000, 60000, 300000, 10000,
					252000, 350000, 260000, 510000,
					2000, 5000],
		'Emission': [300, 400, 1230, 300, 400, 104,
					632, 221, 142, 267],
		'Performance': [60, 88, 90, 87, 83, 81, 72,
						91, 90, 93],
		'Mileage': [76, 89, 89, 57, 79, 84, 78, 99,
					97, 99]
		}

# Creating dataset
df = pd.DataFrame(data,columns=['Price', 'Distance',
								'Emission','Performance',
								'Mileage'])

# Creating a new column in the dataframe that holds
# the Mahalanobis distance for each row
df['calculateMahalanobis'] = calculateMahalanobis(y=df, data=df[[
'Price', 'Distance', 'Emission','Performance', 'Mileage']])

# Display the dataframe
print(df)
