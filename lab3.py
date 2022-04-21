import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from statistics import mean 
import math

# wifi_localization() takes three arguments: 
#    online_file  (string) - name of data file for online phase
#    offline_file (string) - name of data file for offline phase
#    ap_locations (string) - name of data file for access point locations
# It returns a list with two tuple values corresponding to the start and end coordinates of the walk
def wifi_localization(online_file, offline_file, ap_locations):
    # implement your algorithm here
    fingerprints = pd.read_csv(online_file)
    walk = pd.read_csv(offline_file)
    
    #make a list of tuples of the form [(x,y,alpha,SSID,RSSI)] from offline phase data file
    #the final result is stored at recorded_data
    subset = walk[['x', 'y', 'alpha','SSID','RSSI']]
    off_phase_tuples = [tuple(x) for x in subset.to_numpy()]  
    recorded_data = [off_phase_tuples[i:i+3] for i in range(0, len(off_phase_tuples), 3)]
  
      
    #make a list of tuples of the form [(SSID,Time,RSSI)] from online phase data file
    #then group the tuples if they have matching timestamp
    #the final result is stored at observed_data
    subset = fingerprints[['SSID','Time','RSSI']]
    onl_phase_tuples = [tuple(x) for x in subset.to_numpy()]      
    observed_data={}
    for x, y, z in onl_phase_tuples:
        if y in observed_data:
            observed_data[y].append((x,y,z))
        else:
            observed_data[y] = [(x,y,z)]    
    observed_data = list(observed_data.values())
    
    #data recorded at the starting point (online phase)
    ssA_start = observed_data[0][0][2]
    ssB_start = observed_data[0][2][2]
    ssC_start = observed_data[0][1][2]
    
    #data recorded at the starting point (online phase)
    ssA_end = observed_data[len(observed_data) - 1][0][2]
    ssB_end = observed_data[len(observed_data) - 1][2][2]
    ssC_end = observed_data[len(observed_data) - 1][1][2]

    
    return [three_nearest_neighbors(recorded_data,ssA_start, ssB_start, ssC_start), three_nearest_neighbors(recorded_data,ssA_end, ssB_end, ssC_end)] # Your return value should be in this format: [(x_start, y_start), (x_end, y_end)]

#three_nearest_neighbors() take 4 arguments:
#recorded_data (list)          - dataset recorded during offline phase
#A             (integer/float) - signal strength at base station A (online phase)
#B             (integer/float) - signal strength at base station B (online phase)
#C             (integer/float) - signal strength at base station C (online phase)

# it returns tuple (x_value,y_value) approximated using k-nearest-neighbors with k = 3
def three_nearest_neighbors(recorded_data, A, B, C):
    distance_array = []
    for i in range(len(recorded_data)):
        ssA_prime = recorded_data[i][0][4]
        ssB_prime = recorded_data[i][1][4]
        ssC_prime = recorded_data[i][2][4]
        a = np.array((A,B,C))
        b = np.array((ssA_prime,ssB_prime,ssC_prime))
        distance = np.linalg.norm(a-b)
        distance_array.append(distance)
    # get the 3 shortest distance
    A = np.array(distance_array)
    k = 3
    indexes = np.argpartition(A, k)    
    res = A[indexes[:3]]
    I1 = distance_array.index(res[0])
    I2 = distance_array.index(res[1])
    I3 = distance_array.index(res[2])
    res1 = recorded_data[I1]
    res2 = recorded_data[I2]
    res3 = recorded_data[I3]
    x_aver = (res1[0][0] + res2[1][0] + res3[2][0])/3
    y_aver = (res1[0][1] + res2[1][1] + res3[2][1])/3
    return (x_aver, y_aver)
# evaluate() takes two arguments:
#    calculated - estimated start and end coordinates of a walk
#    expected - ground truth
# It returns the score for a walk, 80% of score is based on direction deviation error, 20% is based on location deviation error
def evaluate(calculated, expected):
    X = 0
    Y = 1
    S = 0
    E = 1
    calculated = np.array(calculated)
    expected = np.array(expected)

    # 1. Calculate deviation of walking direction from ground truth
    def get_deviation(calculated, expected):
        calculated = np.array(calculated[E] - calculated[S])
        expected = np.array(expected[E] - expected[S])
        with np.errstate(divide='ignore', invalid='ignore'): 
            dot_prod = np.dot(calculated, expected) / np.linalg.norm(calculated) / np.linalg.norm(expected)
            deviation = np.nan_to_num(np.degrees(np.arccos(dot_prod)), nan=90)
            return deviation if deviation <= 90 else abs(deviation - 180)

    delta_theta = get_deviation(calculated, expected)

    # You will receive full points if deviation <= 30 degrees. 
    # You will receive 0 points if deviation >= 60 degrees.
    # Points for deviation between 30 and 60 degrees will be scaled proportionally.
    theta_score = 1 if(delta_theta <= 30) else max((1 - abs(delta_theta - 30)/30), 0)

    # 2. Calculating absolute distance between calculated and expected S/E coordinates.
    dist_errors = expected - calculated
    s_dist = np.linalg.norm(dist_errors[S], ord=2)
    e_dist = np.linalg.norm(dist_errors[E], ord=2)

    # You will receive full points if error <= 0.5 units. 
    # You will receive 0 points if error >= 3 units.
    # Points for error between 0.5 and 3 units will be scaled proportionally.
    s_score = 1 if(abs(s_dist) <= 0.5) else max(1 - abs(s_dist - 0.5)/2.5, 0)
    e_score = 1 if(abs(e_dist) <= 0.5) else max(1 - abs(e_dist - 0.5)/2.5, 0)
    dist_score = (s_score + e_score) / 2

    return theta_score * 0.8 + dist_score * 0.2

estimated = wifi_localization("walk.csv", "offline.csv", "ap_locations.csv")
expected = [(0., 2),(3.5, 2.5)]
grade = evaluate(estimated, expected)
