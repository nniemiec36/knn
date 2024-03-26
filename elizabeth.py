# want to create a dataframe with states in alphabetical order
# for the year 2000, use cross-fold validation to choose the best value of k for a KNN classifier (there can be many best values of k depending on accuracy, precision, and recall)

# we need to know which party ("party_detailed") won each state in a given year 
# # currently it has raw number of votes per candidate --> classifier label
# use this demographic data as features for each state
# percent of population identified as male (TOT_MAL, TOT_POP)
# percent of population identified as female (TOT_FEMALE, TOT_POP)
# percent of population identified as white (WA_MALE, WA_FEMALE, TOT_POP)
# percent of population identified as black (Black, TOT_POP)
# percent of population identified as hispanic (Hispanic, TOT_POP)

# merge the two datasets into one dataframe. state names are what they have in common, but be careful with upper and lowercase, and make sure you put them in alphabetical order before creating our model

import csv
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

file_path1 = "1976-2020-president.tab"
df1 = pd.read_csv(file_path1,sep='\t').filter(['year', 'state', 'party_detailed', 'candidatevotes', 'totalvotes'], axis=1)
df1.loc[df1['party_detailed'] == 'DEMOCRATIC-FARMER-LABOR', 'party_detailed'] = 'DEMOCRAT' 

# Group by year, state, and party, sum candidatevotes
party_votes_state = df1.groupby(['year', 'state', 'party_detailed'])['candidatevotes'].sum().reset_index()

# Find the index of the maximum candidate votes for each year and state
idx_state = party_votes_state.groupby(['year', 'state'])['candidatevotes'].idxmax()

# Get the corresponding party for each year and state
winning_parties_state = party_votes_state.loc[idx_state, ['year', 'state', 'party_detailed']]

# print(winning_parties_state)

year_data = winning_parties_state[winning_parties_state['year'] == 2000].copy()
# print(year_data)

# print(df1)

file_path2 = "demographics.csv"
df2 = pd.read_csv(file_path2)
df2 = df2.apply(lambda x: x.astype(str).str.upper()).filter(['STNAME','TOT_POP', 'TOT_MALE', 'TOT_FEMALE', 'WA_MALE', 'WA_FEMALE', 'Black', 'Hispanic'], axis=1)
df2['PER_MALE'] = (df2['TOT_MALE'].astype(float)/df2['TOT_POP'].astype(float)) * 100
df2['PER_FEMALE'] = (df2['TOT_FEMALE'].astype(float)/df2['TOT_POP'].astype(float)) * 100
df2['PER_WHITE'] = ((df2['WA_MALE'].astype(float) + df2['WA_FEMALE'].astype(float))/df2['TOT_POP'].astype(float)) * 100
df2['PER_BLACK'] = (df2['Black'].astype(float)/df2['TOT_POP'].astype(float)) * 100
df2['PER_HIS'] = (df2['Hispanic'].astype(float)/df2['TOT_POP'].astype(float)) * 100
df3 = df2.filter(['STNAME','TOT_POP', 'PER_MALE', 'PER_FEMALE', 'PER_WHITE', 'PER_BLACK', 'PER_HIS'], axis=1)
# print(df3)

# features - demographic data that will predict the target variable (y)
x = df2.drop(['STNAME', 'TOT_POP'], axis=1)
# labels - the target value
y = year_data['party_detailed']
print(y.to_string())
# splitting the data into training and testing sets for year 2000
# test size means 20% will be used for testing, the remaining 80% will be used for training
# random state is the random seed for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# scaling features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# k values
k_values = range(4, 11)

# scoring metrics
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro']

# Custom scorer to handle zero division for precision
def custom_precision_score(y_true, y_pred):
    try:
        return metrics.precision_score(y_true, y_pred, average='macro')
    except ZeroDivisionError:
        return 0.0

precision_scorer = metrics.make_scorer(custom_precision_score)

# Dictionary to store the precision scores for each k value
precision_scores = {}

# Iterate over each scoring metric

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(knn, x_train_scaled, y_train, cv=kf, scoring=precision_scorer)
    precision_scores[k] = scores.mean()

        # Find the best k value with the highest precision score
best_k = max(precision_scores, key=precision_scores.get)
best_precision = precision_scores[best_k]

print(y_train.to_string())