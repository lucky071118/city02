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
CRIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Crimes2015.csv')
LOCATION_CATEGORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'locCategory.csv')
WEATHER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Weather.csv')
QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'questionNode_2017.csv')
CATEGORY_TYPE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'category.json')

FEATURE_SIZE = 21
RADIUS = 0.1
START_TIME = datetime.datetime(year=2016, month=1, day=1, hour=1)
END_TIME = datetime.datetime(year=2016, month=12, day=31, hour=23)
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
    x_data_set, y_data_set = create_training_data()
    
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


def create_training_data():
    #  Importing dataset
    # Read crime data
    crime_csv = pandas.read_csv(CRIME_FILE)
    crime_data_set = crime_csv.iloc[ : , :].values

    # Read location data
    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_category_data_set = location_category_csv.iloc[ : , : ].values

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

    # create location grid
    location_grid = create_location_grid(location_category_data_set, category_type_dict)
    # pprint.pprint(location_grid)
    # os.system('pause')
    # create input data set
    x_data_set = None
    
    # create positive data
    for crime_data in crime_data_set:
        new_array = create_x_data_set_format(crime_data, location_grid, weather_data_set)
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))

    # create nagetive data
    data_size = crime_data_set.shape[0]
    location_list = create_location_list(crime_data_set, question_data_set)
    for _ in range(data_size):
        random_location, random_time = create_fake_data(location_list, crime_data_set)
        random_time_str = datetime.datetime.strftime(random_time,'%m/%d/%Y %I:%M:%S %p')
        crime_data = [random_time_str, None, random_location[0], random_location[1]]
        new_array = create_x_data_set_format(crime_data, location_grid, weather_data_set)
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

def create_x_data_set_format(crime_data, location_grid, weather_data_set,):
    
    data_time_str = crime_data[0] #'06/02/2016 07:28:00 PM'
    data_time = datetime.datetime.strptime(data_time_str, '%m/%d/%Y %I:%M:%S %p')
    
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
    b = datetime.datetime.now()
    # Location Category
    location_category_list = find_location_category(latitude, longitude, location_grid)
    c = datetime.datetime.now()
    print(location_category_list)
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
    
    print('all',c-b)
    print('='*20)
    # pprint.pprint(new_array)
    return new_array

def compute_location_grid_index(latitude, longitude):
    latitude_index = int((latitude - MIN_LATITUDE)/GRID_LENGTH)
    longitude_index = int((longitude - MIN_LONGITUDE)/GRID_LENGTH)
    
    return latitude_index, longitude_index



def create_location_grid(location_category_data_set, category_type_dict):
    location_grid = list()
    for _ in range(GRID_NUMBER):
        row_array = list()
        for __ in range(GRID_NUMBER):
            row_array.append(list())
        location_grid.append(row_array)
    
    for location_data in location_category_data_set:
        latitude = location_data[0]
        longitude = location_data[1]
        latitude_index, longitude_index = compute_location_grid_index(latitude, longitude)
        category_type = get_location_category_type(location_data[2], category_type_dict)
        location_data[2] = category_type
        location_grid[latitude_index][longitude_index].append(location_data)
    
    
    return location_grid

def find_location_category(latitude, longitude, location_grid):
    result_array = []
    latitude_index, longitude_index = compute_location_grid_index(latitude, longitude)
    start_location = (latitude, longitude)
    a = datetime.datetime.now()
    for i in range(-1,2):
        for j in range(-1,2):
            result_array.extend(find_location_in_grid(start_location, latitude_index+i, longitude_index+j, location_grid))
    b = datetime.datetime.now()
    print('b-a',b-a)
    return list(set(result_array))

def find_location_in_grid(start_location, latitude_index, longitude_index, location_grid):
    result_array = []
    if 0 < latitude_index < GRID_NUMBER:
        if 0 < longitude_index < GRID_NUMBER:
            specific_location_grid = location_grid[latitude_index][longitude_index]
            print(len(specific_location_grid))
            for location_category_data in specific_location_grid:
                end_latitude =  location_category_data[0]
                end_longitude =  location_category_data[1]
                end_location = (end_latitude, end_longitude)
                miles = distance.distance(end_location, start_location).miles
                
                if miles < RADIUS:
                    result_array.append(location_category_data[2])
    return result_array


def find_weather(data_time, weather_data_set):
    new_date_time = data_time.replace(minute=0)
    if data_time.minute > 30:
        new_date_time +=  datetime.timedelta(hours=1)
    # data_time datetime.datetime.strptime(data_time, '%Y-%m-%d %H:%M:%S')
    for weather_data in weather_data_set:
        weather_time_str = weather_data[0] #2016-01-01 01:00:00
        weather_time = datetime.datetime.strptime(weather_time_str, '%Y-%m-%d %H:%M:%S')
        if weather_time == new_date_time:
            return weather_data[1:]







def create_location_list(crime_data_set, question_data_set):
    location_list = []
    for crime_data in crime_data_set:
        location = (crime_data[2], crime_data[3])
        location_list.append(location)
            
    for question_data in question_data_set:
        location = (question_data[0], question_data[1])
        location_list.append(location)
    return list(set(location_list))

def create_fake_data(location_list, crime_data_set):
    equal = True
    
    
    while equal:
        random_location = random.choice(location_list)
        random_time = random_date_time()
        equal = False
        for crime_data in crime_data_set:
            data_time_str = crime_data[0] #'06/02/2016 07:28:00 PM'
            data_time = datetime.datetime.strptime(data_time_str, '%m/%d/%Y %I:%M:%S %p')
            crime_location = (crime_data[2], crime_data[3])
            if random_location == crime_location:
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
    main()
    # create_y_data(20)