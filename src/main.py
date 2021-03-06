import os
import random
import datetime
import json
import random
import pprint
import configparser
import csv

import numpy
import pandas
from geopy import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from location_category import get_location_category_type
from location_category import get_location_feature_size
from location_category import encode_single_location_category_list
from location_category import encode_multiple_location_category_list
from location_category import no_location_feature
from weather_description_category import get_weather_description_category
from weather_description_category import encode_weather_category_list
from weather_description_category import get_weather_feature_size




DIR_PATH = 'data'
CRIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Crimes2016.csv')
LOCATION_CATEGORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'locCategory.csv')
WEATHER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Weather.csv')
QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'questionNode_2017.csv')
CATEGORY_TYPE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'category.json')







START_TIME = datetime.datetime(year=2016, month=1, day=1, hour=1)
END_TIME = datetime.datetime(year=2016, month=12, day=31, hour=23)
MISSING_WEATHER_DATA_TIME = datetime.datetime(year=2016, month=1, day=1, hour=0)
START_DATE = datetime.date(year=2016, month=1, day=1)
INVAILD_TIME = datetime.datetime(year=2017, month=12, day=1)
VAILD_TIME = datetime.datetime(year=2017, month=11, day=29)
MAX_LATITUDE = 42.0436
MIN_LATITUDE = 41.6236
GRID_LENGTH = 0.07
MAX_LONGITUDE = -87.5117
MIN_LONGITUDE = -87.9317
GRID_NUMBER = 6



config = configparser.ConfigParser()
config.read('setting.config')
feature_config = config['FEATURE']
KNN_NUMBER = int(config['OTHER']['k_nearest_neighbor'])
TRAIN_DATA_SIZE_SUBSET = int(config['OTHER']['train_data_size_subset'])
RADIUS = float(config['OTHER']['radius'])
MULTIPLE_LOCATION_TYPE = config['OTHER'].getboolean('multiple_location_type')
GET_RESULT = config['OTHER'].getboolean('get_result')


def main():
    feature_size = check_feature()
    print('='*10, 'Preprocessing', '='*10)
    time_a = datetime.datetime.now()
    x_data_set, y_data_set, predict_data_set_list = create_data(feature_size)
    
    # Splitting the dataset into the Training set and Test set
    result = train_test_split(x_data_set, y_data_set, test_size=0.25, random_state=0)
    x_train, x_test, y_train, y_test = result

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    new_predict_data_set_list = []
    for predict_data_set in predict_data_set_list:
        new_predict_data_set = sc.transform(predict_data_set)
        new_predict_data_set_list.append(new_predict_data_set)
    print('='*10, 'Training', '='*10)
    time_b = datetime.datetime.now()

    # Fitting SVM to the Training set

    # classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    # classifier = SVC()
    # classifier = SVC()
    # classifier = LogisticRegression()
    # classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5)
    classifier = MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=(500,500), random_state=1)

    # Train
    classifier.fit(x_train, y_train)
    # save the model to disk
    # filename = 'finalized_model.sav'
    # joblib.dump(classifier, filename)
    
    # export_graphviz(classifier, out_file="tree.dot", feature_names=['month', 'hour','week'])

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix', cm)
    (true_negative, false_positive, false_negative, true_positive) = numpy.ravel(cm)
    total = true_negative + false_positive + false_negative + true_positive
    accuracy = (true_negative + true_positive)/total
    precision = true_positive / (false_positive + true_positive)
    recall = true_positive / (false_negative + true_positive)
    f_score = 2 * precision * recall / ( precision + recall )
    
    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =',recall)
    print('f_score =',f_score)
    time_c = datetime.datetime.now()
    print('Preprocessing Time is',time_b-time_a)
    print('Training Time is',time_c-time_b)
    print('Total Time is',time_c-time_a)

    if GET_RESULT:
        compute_result(classifier, new_predict_data_set_list)
        

def compute_result(classifier, predict_data_set_list):
    # Read question data
    question_csv = pandas.read_csv(QUESTION_FILE)
    question_data_set = question_csv.iloc[ : , : ].values
    
    
    for index, predict_data_set in enumerate(predict_data_set_list):
        result_set = classifier.predict(predict_data_set)
        result_answer = 0
        if any(result_set):
            result_answer = 1
        question_data_set[index,4] = result_answer
    write_result_file(question_data_set)     
            
def write_result_file(question_data_set):
    
    with open('result.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Field
        writer.writerow(['Latitude','Longitude','Date','Time slot','prediction'])
        
        # value
        writer.writerows(question_data_set)

def check_feature():
    print('='*10 + 'Feature' + '='*10)
    feature_size = 0
    feature_location_size = get_location_feature_size(feature_config)
    feature_size += feature_location_size
    for feature in feature_config:
        if 'location' not in feature:
            if 'weather_description' == feature:
                if feature_config.getboolean(feature):
                    print(feature)
                    feature_size += get_weather_feature_size()
            else:
                if feature_config.getboolean(feature):
                    print(feature)
                    feature_size += 1
    print('Total Feature Size :', feature_size)
    return feature_size

def create_data(feature_size):
    #  Importing dataset
    # Read crime data
    # crime_data_index_list = range(0,266584,TRAIN_DATA_SIZE_SUBSET)
    crime_data_index_list = []
    data_size = int(266584/TRAIN_DATA_SIZE_SUBSET)
    for index in range(0, data_size):
        start = index*TRAIN_DATA_SIZE_SUBSET
        end = (index+1)*TRAIN_DATA_SIZE_SUBSET
        choice_number = random.choice(range(start,end))
        crime_data_index_list.append(choice_number)

    

    crime_csv = pandas.read_csv(CRIME_FILE)
    crime_data_set = crime_csv.iloc[ crime_data_index_list , :].values


    # Read all crime data
    crime_csv = pandas.read_csv(CRIME_FILE)
    all_crime_data_set = crime_csv.iloc[ : , :].values

    # Read location data
    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_category_data_set = location_category_csv.iloc[ : , : ].values
    
    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_data_set = location_category_csv.iloc[ : , [0,1] ].values
    
    # Read weather data
    weather_csv = pandas.read_csv(WEATHER_FILE)
    weather_data_set = weather_csv.iloc[ : , : ].values

    # Read question data
    question_csv = pandas.read_csv(QUESTION_FILE)
    question_data_set = question_csv.iloc[ : , : ].values

    #  Read category type data
    category_type_dict = None
    with open(CATEGORY_TYPE_FILE, 'r',encoding = "utf-8") as f:
        category_type_dict = json.load(f)['response']['categories']

    
    # Handling the missing data
     
    imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
    imputer = imputer.fit(weather_data_set[ : , [1,2,3]])
    weather_data_set[ : , [1,2,3]] = imputer.transform(weather_data_set[ : , [1,2,3]])
    
    
    # analysis location category
    analysis_location_category(location_category_data_set, category_type_dict)
    

    # create KDTree
    kd_tree = KDTree(location_data_set, leaf_size=60, metric='euclidean')

    # create x data set
    x_data_set = None

    a =0
    # create positive data
    for crime_data in crime_data_set:
        a+=1
        if a%1000 ==0:
            print('='*10 + str(a) + '='*10)
        new_array = create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set, feature_size)
        
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))

    # create nagetive data
    data_size = crime_data_set.shape[0]
    location_list = create_location_list(all_crime_data_set, question_data_set)
    crime_dict = create_crime_dict(all_crime_data_set)
    for _ in range(data_size):
        a+=1
        if a%1000 ==0:
            print('='*10 + str(a) + '='*10)
        random_location, random_time = create_fake_data(location_list, crime_dict)
        random_time_str = datetime.datetime.strftime(random_time,'%m/%d/%Y %I:%M:%S %p')
        crime_data = [random_time_str, "123", random_location[0], random_location[1]]
        new_array = create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set, feature_size)
        
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))

    # create y data set
    y_data_set = create_y_data(data_size)
    print('='*10, 'predict_data', '='*10)
    # create predict data set
    predict_data_set_list = []
    if GET_RESULT:
        changed_crime_data_set = change_question_to_crime(question_data_set)
        for crime_data_list in changed_crime_data_set:
            predict_data_set = None
            for crime_data in crime_data_list:
                new_array = create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set, feature_size)
        
                if predict_data_set is None:
                    predict_data_set = new_array
                else:
                    predict_data_set = numpy.concatenate((predict_data_set, new_array))

            predict_data_set_list.append(predict_data_set)
        print('predict_data_set_list',len(predict_data_set_list))

    return x_data_set, y_data_set, predict_data_set_list
    


def create_y_data(data_size):
    
    a = numpy.ones(data_size)
    b= numpy.zeros(data_size)
    y_data_set = numpy.concatenate((a, b), axis=None)
    return y_data_set

def create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set, feature_size):
    
    data_time_str = crime_data[0] #'06/02/2016 07:28:00 PM'
    data_time = datetime.datetime.strptime(data_time_str, '%m/%d/%Y %I:%M:%S %p')
    data_time = data_time.replace(second=0)
    # Month
    month = int(data_time.strftime('%m'))

    # Day
    day = int(data_time.strftime('%d'))
    
    # Hour
    hour = int(data_time.strftime('%H'))
    
    # Week
    week = data_time.weekday() + 1
    
    # Latitude
    latitude = round(crime_data[2],8)
    

    # Longitude
    longitude = round(crime_data[3],8)
    

    
    
    
    # Location Category
    location_category_list = find_nearest_neighbors(latitude, longitude, kd_tree, location_category_data_set)
    if MULTIPLE_LOCATION_TYPE:
        location_category_encoding_list = encode_multiple_location_category_list(location_category_list)
    else:
        location_category_encoding_list = encode_single_location_category_list(location_category_list)
    
    weather_array = find_weather(data_time, weather_data_set)
    # array is [humidity,pressure,temperature,weather_description,wind_direction,wind_speed]
    
    # Humidity
    humidity = round(weather_array[0],1)

    # Pressure
    pressure = round(weather_array[1],1)

    # Temperature
    temperature = round(weather_array[2],2)

    # Weather Description
    weather_description = weather_array[3]
    weather_category_list = get_weather_description_category(weather_description)
    weather_category_encoding_list = encode_weather_category_list(weather_category_list)

    # Wind Direction
    wind_direction = round(weather_array[4],1)

    # Wind Speed
    wind_speed = round(weather_array[5],1)

    
    # create new array
    new_array = numpy.zeros((1, feature_size))

    # put features to new array
    
    array_index = 0
    for feature_name in feature_config:
        if 'location' in feature_name:
            continue
        if feature_name == 'weather_description':
            continue
        if feature_config.getboolean(feature_name):
            new_array[0,array_index] = locals()[feature_name]
            array_index += 1

   
    if not no_location_feature(feature_config):
        for location_category in location_category_encoding_list:
            new_array[0,array_index] = location_category
            array_index += 1

    if feature_config.getboolean('weather_description'):
        for weather_category in weather_category_encoding_list:
            new_array[0,array_index] = weather_category
            array_index += 1
    
    return new_array

def find_nearest_neighbors(latitude, longitude, kd_tree, location_category_data_set):
    location_category_list = []
    distance_list, index_list = kd_tree.query(numpy.array([[latitude,longitude]]), k=KNN_NUMBER)
    for i, array_index in enumerate(index_list[0]):
        if distance_list[0][i] < RADIUS:
            location_category_list.append(location_category_data_set[array_index][2])
    return location_category_list

def analysis_location_category(location_category_data_set, category_type_dict):
    for location_data in location_category_data_set:
        category = get_location_category_type(location_data[2], category_type_dict)
        location_data[2] = category


def find_weather(data_time, weather_data_set):

    if MISSING_WEATHER_DATA_TIME <= data_time <  START_TIME:
        data_time = START_TIME

    new_date_time = data_time.replace(minute=0, second=0, microsecond=0)
    if data_time.minute > 30:
        new_date_time +=  datetime.timedelta(hours=1)
    # data_time datetime.datetime.strptime(data_time, '%Y-%m-%d %H:%M:%S')
    index = compute_weather_index(new_date_time)
    return weather_data_set[index,1:]

def compute_weather_index(new_date_time):
    index = 0
    time_delta = new_date_time - START_TIME
    index += (time_delta.days*24)
    index += (time_delta.seconds/3600)
    
    return int(index)






def create_crime_dict(crime_data_set):
    crime_dict = {}
    for crime_data in crime_data_set:
        location = (crime_data[2],crime_data[3])
        crime_dict.setdefault(location, []).append(crime_data[0])
    
    return crime_dict



def create_location_list(crime_data_set, question_data_set):
    location_list = []
    for crime_data in crime_data_set:
        location = (round(crime_data[2],8), round(crime_data[3],8))
        location_list.append(location)
            
    for question_data in question_data_set:
        location = (round(question_data[0],8), round(question_data[1],8))
        location_list.append(location)
    return list(set(location_list))

def create_fake_data(location_list, crime_dict):
    equal = True
    
    
    while equal:
        random_location = random.choice(location_list)
        random_time = random_date_time()
        equal = False
        data_time_str_list = crime_dict.get(random_location,[])
        if data_time_str_list:
            for data_time_str in data_time_str_list:
                #'06/02/2016 07:28:00 PM'
                data_time = datetime.datetime.strptime(data_time_str, '%m/%d/%Y %I:%M:%S %p')
                if data_time.date() == random_time.date():
                    delta = abs(data_time - random_time)
                    if delta < datetime.timedelta(hours=6):
                        equal = True
    return random_location, random_time

def random_date_time():
    random_number = random.random() #[0,1)
    random_time = START_TIME + (END_TIME - START_TIME)*random_number
    return random_time.replace(second=0, microsecond=0)


def change_question_to_crime(question_data_set):
    changed_crime_data_set = []
    time_slot_dict = {
        'midnight':0,
        'morning':6,
        'afternoon':12,
        'night':18
    }
    for question_data in question_data_set:
        data_time_str = question_data[2]
        data_date = datetime.datetime.strptime(data_time_str, '%Y/%m/%d')

        if data_date > INVAILD_TIME:
            data_date = VAILD_TIME

        latitude = question_data[0]
        longitude = question_data[1]
        time_slot = question_data[3]
        start_hour = time_slot_dict[time_slot]
        crime_data_list =[]
        for index in range(6):
            data_hour = start_hour+index
            data_time = data_date + datetime.timedelta(hours=data_hour)
            data_time_str = datetime.datetime.strftime(data_time,'%m/%d/%Y %I:%M:%S %p')
            crime_data = [data_time_str, "123", latitude, longitude]
            crime_data_list.append(crime_data)
                
        changed_crime_data_set.append(crime_data_list)
    return changed_crime_data_set
        



if __name__ == '__main__':
    # test = datetime.datetime(year=2016, month=1, day=2, hour=1)
    # index = compute_weather_index(test)
    # print(index)
    main()
    # create_y_data(20)