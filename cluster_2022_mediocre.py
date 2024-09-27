import pandas as pd
players20 = pd.read_excel('players_final_22_mediocre.xlsx')
#players201 = pd.read_excel('players_final_21_1.xlsx')
corr20 = players20.corr(numeric_only=True)
players20 = players20[players20['ACTUAL_MINUTES']>500]

players20.columns

avg_features = [ 'ACTUAL_MINUTES','DISPLAY_FI_LAST','GP','FG_PCT','FG3_PCT',  'FT_PCT', 'AVG_TOT_REB',
         'AVG_AST', 'AVG_STL',
       'AVG_TURNOVERS', 'AVG_BLK', 'AVG_PTS',  
       
       'percentagePointsMidrange2pt',
       'percentagePointsFastBreak',
       'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
       'percentagePointsPaint', 'percentageAssisted2pt',
       'percentageUnassisted2pt', 'percentageAssisted3pt',
       'percentageUnassisted3pt',  
       'playerPoints',  
        'matchupFieldGoalPercentage','PIE', 'PER'] #,'OWS','DWS','OBPM', 'DBPM']
corr20 = players20[avg_features].corr(numeric_only=True)


corr20 = players20[avg_features].corr(numeric_only=True)
players20 = players20[avg_features]

import seaborn as sns
#sns.histplot(data=players20, x='GP')
players20 = players20[players20['GP']>25]

# Perform standard scaling on the selected features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = [ 'ACTUAL_MINUTES','FG_PCT','FG3_PCT',  'FT_PCT', 'AVG_TOT_REB',
         'AVG_AST', 'AVG_STL',
       'AVG_TURNOVERS', 'AVG_BLK', 'AVG_PTS',  
       
       'percentagePointsMidrange2pt',
       'percentagePointsFastBreak',
       'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
       'percentagePointsPaint', 'percentageAssisted2pt',
       'percentageUnassisted2pt', 'percentageAssisted3pt',
       'percentageUnassisted3pt',  
       'playerPoints', 
        'matchupFieldGoalPercentage','PIE', 'PER']#,'OWS','DWS','OBPM', 'DBPM']
players20[features] = scaler.fit_transform(players20[features])

scaled_features = players20[features]


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score

silhouette_scores = []
k_values = range(2, 15)  # You can adjust the range as needed

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, init="random")
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()



#elbow method

inertia_values = []
k_values = range(1, 15)  # You can adjust the range as needed

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


    
    
data = []
for i in range(len(inertia_values) - 1):
    in_new = inertia_values[i ] - inertia_values[i+1]
    data.append(in_new)

dfg = pd.DataFrame(data, columns=['Values'], index=range(1, len(data) + 1))
sorted_dfg = dfg.sort_values(by='Values', ascending=False)



k =6



# Perform KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=1,init='random')
players20['cluster'] = kmeans.fit_predict(scaled_features)




cluster_centers = kmeans.cluster_centers_
cluster_centers_df_mediocre = pd.DataFrame(cluster_centers, columns=features)
print(cluster_centers_df_mediocre)

import matplotlib.pyplot as plt
import seaborn as sns

# Define a custom color palette with distinct colors for each cluster
# You can choose any color scheme you prefer
custom_palette = sns.color_palette('hsv', n_colors=k)

# Plot clusters using two selected features (you can choose different features)
sns.scatterplot(x='AVG_TOT_REB', y='AVG_PTS', hue='cluster', data=players20, palette=custom_palette)
plt.title('KMeans Clustering2022')
plt.xlabel('Field Goal Percentage')
plt.ylabel('Points')
plt.show()

cluster_players_mediocre = players20[['DISPLAY_FI_LAST', 'cluster']]



