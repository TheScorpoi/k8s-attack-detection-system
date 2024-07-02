import pandas as pd


data1 = pd.read_csv('../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/elastic_february2022_data.csv')
data2 = pd.read_csv('../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/elastic_may2022_data.csv', delimiter=';')

data2 = data2.drop('_source_network_packets', axis=1)

sampled_data2 = data2.sample(frac=0.1)

combined_data = pd.concat([data1, sampled_data2], ignore_index=True)

combined_data.to_csv('../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/concat_fev_30_may.csv', index=False)