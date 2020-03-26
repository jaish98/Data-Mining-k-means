import pandas as pd # learn more: https://python.org/pypi/pandas
import numpy as np
import random
import math
import copy

def calculateMeanDistance(final_df,centroid):
    value_l = []
    for index, row in final_df.iterrows():
        value = euclideanDistance([row['gender'],row['birth_date'],row['weight']],centroid)
        value_l.append(value)
        
    return sum(value_l)/len(value_l)

def euclideanDistance(p1,p2):
    value = pow((p1[0] - p2[0]),2) + pow((p1[1] - p2[1]),2) + pow((p1[2] - p2[2]),2)
    return math.sqrt(value)

def newCentroid(new_df,new_coor_centroid):
    coor_centroid = []
    for x in range(3):
        sum_gender = 0
        sum_birth_date = 0
        sum_weight = 0
        filtered_df = new_df[new_df['centroid'] == x]
        
        if filtered_df.shape[0] != 0:
            for index, row in filtered_df.iterrows():
                sum_gender += row['gender']
                sum_birth_date += row['birth_date']
                sum_weight += row['weight']

            if filtered_df.shape[0] == 0:
                print(filtered_df)

            mean_gender = sum_gender/filtered_df.shape[0]
            mean_birth_date = sum_birth_date/filtered_df.shape[0]
            mean_weight = sum_weight/filtered_df.shape[0]
            coor = [mean_gender,mean_birth_date,mean_weight]
        else:
            coor = new_coor_centroid[x]
        coor_centroid.append(coor)
        
    return coor_centroid


df = pd.read_csv('https://raw.githubusercontent.com/machine-learning-course/syllabus/gh-pages/hiw-2019b/dataset-students-ml-2019b.csv')
df['nim'] = df['nim'].astype(str)

# TODO: Your k-means algorithm here

new_df = df[['gender', 'birth_date', 'weight']].copy()

for index, row in new_df.iterrows():
    split_date = row['birth_date'].split('-')
    new_df.loc[index,'birth_date'] = split_date[0]
    
new_df.loc[new_df['gender'] == 'Male', 'gender'] = 1
new_df.loc[new_df['gender'] == 'Female', 'gender'] = 2

new_df = new_df.astype({"gender":'int64', "birth_date":'int64'})

new_df["centroid"] = 0

initial_centroid = []
for x in range(3):
    value = random.randint(0, new_df.shape[0]-1)
    while value in initial_centroid:
        value = random.randint(0, new_df.shape[0]-1)

    initial_centroid.append(value)

for index, row in new_df.iterrows():
    dist_l = []
    p1 = [row['gender'],row['birth_date'],row['weight']]
    for x in range(3):
        p2 = [new_df.loc[initial_centroid[x],'gender'],new_df.loc[initial_centroid[x],'birth_date'],new_df.loc[initial_centroid[x],'weight']]
        value = pow((p1[0] - p2[0]),2) + pow((p1[1] - p2[1]),2) + pow((p1[2] - p2[2]),2)
        dist_l.append(math.sqrt(value))
      
    row['centroid'] = np.argmin(dist_l)
    
new_coor_centroid = []
for x in range(3):
    point = [new_df.loc[initial_centroid[x],'gender'],new_df.loc[initial_centroid[x],'birth_date'],new_df.loc[initial_centroid[x],'weight']]
    new_coor_centroid.append(point)

while True:
    new_centroid = newCentroid(new_df,new_coor_centroid)
    new_coor_centroid = copy.deepcopy(new_centroid)
    prev_df = copy.deepcopy(new_df)
    for index, row in new_df.iterrows():
        dist_l = []
        p1 = [row['gender'],row['birth_date'],row['weight']]
        
        for x in range(3):
            distance = euclideanDistance(p1, new_centroid[x])
            dist_l.append(distance)
            
        row['centroid'] = np.argmin(dist_l)
    
    prev_1 = prev_df[prev_df['centroid'] == 0]
    prev_2 = prev_df[prev_df['centroid'] == 1]
    prev_3 = prev_df[prev_df['centroid'] == 2]
    
    current_1 = new_df[new_df['centroid'] == 0]
    current_2 = new_df[new_df['centroid'] == 1]
    current_3 = new_df[new_df['centroid'] == 2]
    if prev_1.equals(current_1) and prev_2.equals(current_2) and prev_3.equals(current_3):
        break;
            
print('Model Code: D')

for index, row in df.iterrows() :
  centroid_nim = new_df['centroid'][index]
  mean_distance = 0
  if centroid_nim == 0:
    mean_distance = calculateMeanDistance(current_1,new_coor_centroid[0])
  elif centroid_nim == 1:
    mean_distance = calculateMeanDistance(current_2,new_coor_centroid[1])
  elif centroid_nim == 2:
    mean_distance = calculateMeanDistance(current_3,new_coor_centroid[2])

  print('%s,%s,%0.4f' % (row['nim'], centroid_nim, mean_distance))
