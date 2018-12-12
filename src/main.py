import os
import datetime
import json
import random
import numpy
import pandas
import pprint
from geopy import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
from location_category import get_location_category_type

DIR_PATH = 'data'
CRIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Crimes2016.csv')
LOCATION_CATEGORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'locCategory.csv')
WEATHER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Weather.csv')
QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'questionNode_2017.csv')
CATEGORY_TYPE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'category.json')

FEATURE_SIZE = 21
# RADIUS = 0.1
RADIUS = 0.002

START_TIME = datetime.datetime(year=2016, month=1, day=1, hour=1)
END_TIME = datetime.datetime(year=2016, month=12, day=31, hour=23)
MISSING_WEATHER_DATA_TIME = datetime.datetime(year=2016, month=1, day=1, hour=0)
START_DATE = datetime.date(year=2016, month=1, day=1)
MAX_LATITUDE = 42.0436
MIN_LATITUDE = 41.6236
GRID_LENGTH = 0.07
MAX_LONGITUDE = -87.5117
MIN_LONGITUDE = -87.9317
GRID_NUMBER = 6

MAPPING_LIST = [
    "Arts & Entertainment",
    "College & University",
    "Event",
    'Food',
    'Nightlife Spot',
    'Outdoors & Recreation',
    'Professional & Other Places',
    'Residence',
    'Shop & Service',
    'Travel & Transport'
]

# MAPPING_DICT = {
#     "Arts & Entertainment" : 1,
#     "College & University" : 2,
#     "Event":3,
#     'Food':4,
#     'Nightlife Spot':5,
#     'Outdoors & Recreation': 6,
#     'Professional & Other Places':7,
#     'Residence':8,
#     'Shop & Service':9,
#     'Travel & Transport':10
# }

def main():
    a = datetime.datetime.now()
    x_data_set, y_data_set = create_training_data()
    # pprint.pprint(x_data_set)
    # Splitting the dataset into the Training set and Test set
    result = train_test_split(x_data_set, y_data_set, test_size=0.25, random_state=0)
    x_train, x_test, y_train, y_test = result

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Fitting SVM to the Training set
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix', cm)
    total =  cm.item(0) + cm.item(1) + cm.item(2) + cm.item(3)
    accuracy = (cm.item(0) + cm.item(3)) / total
    precision = cm.item(3) / (cm.item(3)+cm.item(1))
    recall = cm.item(3) / (cm.item(2)+cm.item(3))
    print('Accuracy =', accuracy)
    print('Precision =', precision)
    print('Recall =',recall)
    b = datetime.datetime.now()
    print('Time is',b-a)

def create_training_data():
    #  Importing dataset
    # Read crime data
    crime_csv = pandas.read_csv(CRIME_FILE)
    crime_data_set = crime_csv.iloc[ : , :].values

    # Read location data
    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_category_data_set = location_category_csv.iloc[ : , : ].values
    
    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_data_set = location_category_csv.iloc[ : , [0,1] ].values
    
    # Read weather data
    weather_csv = pandas.read_csv(WEATHER_FILE)
    weather_data_set = weather_csv.iloc[ : , : ].values

    # # Read question data
    question_csv = pandas.read_csv(QUESTION_FILE)
    question_data_set = question_csv.iloc[ : , : ].values

    # # Read category type data
    category_type_dict = None
    with open(CATEGORY_TYPE_FILE, 'r',encoding = "utf-8") as f:
        category_type_dict = json.load(f)['response']['categories']

    # analysis location category
    analysis_location_category(location_category_data_set, category_type_dict)
    

    # create KDTree
    kd_tree = KDTree(location_data_set, leaf_size=60, metric='euclidean')

    # create input data set
    x_data_set = None

    a =0
    # create positive data
    for crime_data in crime_data_set:
        a+=1
        if a%1000 ==0:
            print(a)
        new_array = create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set)
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))

    # create nagetive data
    data_size = crime_data_set.shape[0]
    location_list = create_location_list(crime_data_set, question_data_set)
    crime_dict = create_crime_dict(crime_data_set)
    for _ in range(data_size):
        a+=1
        if a%1000 ==0:
            print(a)
        random_location, random_time = create_fake_data(location_list, crime_dict)
        random_time_str = datetime.datetime.strftime(random_time,'%m/%d/%Y %I:%M:%S %p')
        crime_data = [random_time_str, None, random_location[0], random_location[1]]
        new_array = create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set)
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))

    y_data_set = create_y_data(data_size)
    return x_data_set, y_data_set
    # pprint.pprint(x_data_set)

def create_y_data(data_size):
    
    a = numpy.ones(data_size)
    b= numpy.zeros(data_size)
    y_data_set = numpy.concatenate((a, b), axis=None)
    return y_data_set

def create_x_data_set_format(crime_data, kd_tree, location_category_data_set, weather_data_set,):
    
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
    latitude = crime_data[2]

    # Longitude
    longitude = crime_data[3]
    

    
    
    
    # Location Category
    location_category_list = find_nearest_neighbors(latitude, longitude, kd_tree, location_category_data_set)
    location_category_encoding_list = []
    for category_name in MAPPING_LIST:
        if category_name in location_category_list:
            location_category_encoding_list.append(1)
        else:
            location_category_encoding_list.append(0)
    
    weather_array = find_weather(data_time, weather_data_set)
    # array is [humidity,pressure,temperature,weather_description,wind_direction,wind_speed]
    
    # Humidity
    humidity = weather_array[0]

    # Pressure
    pressure = weather_array[1]

    # Temperature
    temperature = weather_array[2]

    # Weather Description
    weather_description = weather_array[3]

    # Wind Direction
    wind_direction = weather_array[4]

    # Wind Speed
    wind_speed = weather_array[5]

    
    
    # create new array
    new_array = numpy.zeros((1, FEATURE_SIZE))

    # put features to new array
    new_array[0,0] = month
    new_array[0,1] = day
    new_array[0,2] = hour
    new_array[0,3] = week
    new_array[0,4] = latitude
    new_array[0,5] = longitude
    
    for index, location_category in enumerate(location_category_encoding_list):
        new_array[0,6+index] = location_category


    new_array[0,16] = humidity
    new_array[0,17] = pressure
    new_array[0,18] = temperature
    new_array[0,19] = wind_direction
    new_array[0,20] = wind_speed
    # pprint.pprint(new_array)
    return new_array

def find_nearest_neighbors(latitude, longitude, kd_tree, location_category_data_set):
    location_category_list = []
    distance_list, index_list = kd_tree.query(numpy.array([[latitude,longitude]]), k=5)
    for i, array_index in enumerate(index_list[0]):
        if distance_list[0][i] < RADIUS:
            location_category_list.append(location_category_data_set[array_index][2])
    # pprint.pprint(list(set(location_category_list)))
    return list(set(location_category_list))

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
        location = (crime_data[2], crime_data[3])
        location_list.append(location)
            
    for question_data in question_data_set:
        location = (question_data[0], question_data[1])
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





if __name__ == '__main__':
    # test = datetime.datetime(year=2016, month=1, day=2, hour=1)
    # index = compute_weather_index(test)
    # print(index)
    main()
    # create_y_data(20)