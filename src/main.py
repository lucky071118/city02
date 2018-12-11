import os
import datetime
import json
import numpy
import pandas
import pprint
from geopy import distance
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DIR_PATH = 'data'
CRIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Crimes2016.csv')
LOCATION_CATEGORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'locCategory.csv')
WEATHER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'Weather.csv')
QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'questionNode_2017.csv')
CATEGORY_TYPE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIR_PATH, 'category.json')

FEATURE_SIZE = 6
RADIUS = 10

def main():
    create_training_data()
    # pprint.pprint(x_data_set)
    


def create_training_data():
    #  Importing dataset
    crime_csv = pandas.read_csv(CRIME_FILE)
    crime_data_set = crime_csv.iloc[ : , :].values

    location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
    location_category_data_set = location_category_csv.iloc[ : , : ].values

    weather_csv = pandas.read_csv(WEATHER_FILE)
    weather_data_set = weather_csv.iloc[ : , : ].values

    category_type_dict = None
    with open(CATEGORY_TYPE_FILE, 'r',encoding = "utf-8") as f:
        category_type_dict = json.load(f)['response']['categories']

    # create input data set
    data_size = crime_data_set.shape[0]
    x_data_set = None

    
    # create positive data
    for index, crime_data in enumerate(crime_data_set):

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

        # Location Category
        location_category_list = find_location_category(
            latitude,
            longitude,
            location_category_data_set,
            category_type_dict
        )

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

        print('location_category_list')
        print(location_category_list)
        print('humidity,pressure,temperature,weather_description,wind_direction,wind_speed')
        print( humidity,pressure,temperature,weather_description,wind_direction,wind_speed)
        # create new array
        new_array = numpy.zeros((1, FEATURE_SIZE))

        # put features to new array
        new_array[0,0] = month
        new_array[0,1] = day
        new_array[0,2] = hour
        new_array[0,3] = week
        new_array[0,4] = latitude
        new_array[0,5] = longitude
        if x_data_set is None:
            x_data_set = new_array
        else:
            x_data_set = numpy.concatenate((x_data_set, new_array))
        
    # create nagetive data
    # pprint.pprint(location_category_dict)
    # old_x_data_set = numpy.copy(x_data_set)
    # for x_data in old_x_data_set:
    #     location = str(x_data[4]) + '&' + str(x_data[5])
    #     print(location)
    #     print(location_category_dict.get(location))
        

    # Splitting the dataset into the Training set and Test set
    # result = train_test_split(x, y, test_size=0.25, random_state=0)
    # x_train, x_test, y_train, y_test = result

def find_location_category(latitude, longitude, location_category_data_set, category_type_dict):
    result_array = []
    start_location = (latitude, longitude)
    for location_category_data in location_category_data_set:
        end_latitude =  location_category_data[0]
        end_longitude =  location_category_data[1]
        end_location = (end_latitude, end_longitude)
        miles = distance.distance(end_location, start_location).miles

        if miles < RADIUS:
            category_type = get_location_category_type(location_category_data[2], category_type_dict)
            result_array.append(category_type)
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

def category_check(location,category_type_dict):
    result = None
    name_dic = {"Mall":"Shop & Service",
                "Subway":"Food",
                "Athletic & Sport":"Outdoors & Recreation",
                "Hiking Trail":"Outdoors & Recreation",
                "Car Dealership":"Shop & Service",
                "Light Rail":"Travel & Transport",
                "Frozen Yogurt":"Food",
                "Stable":"Outdoors & Recreation",
    }
    if location in name_dic.keys():
        result = name_dic[location]
       
    for high_category in category_type_dict:
        if location == high_category["name"]:
            result = high_category["name"]
        if high_category["categories"] != []:
            for mid_category in high_category["categories"]:
                if location == mid_category["name"]:
                    result = high_category["name"]
                if mid_category["categories"] != []:
                    for low_category in mid_category["categories"]:
                        if location == low_category["name"]:
                            result = high_category["name"]
                        if low_category["categories"] != []:
                            for last_category in low_category["categories"]:
                                if location == last_category["name"]:
                                    result = high_category["name"]
                                if last_category["categories"] != []:
                                    for end_category in last_category["categories"]:
                                        if location == end_category["name"]:
                                            result = high_category["name"]
    return result

def get_location_category_type(location, category_type_dict):
    result = category_check(location,category_type_dict)
    location_list = []
    result_list = []
    if result is None:
        if "/" in location:
            location_list = location.split("/")
            for single_location in location_list:
                single_location = single_location.strip()
                new_result = category_check(single_location,category_type_dict)
                result_list.append(new_result)
        else:
            result = "Food"

        if result_list:
            result = result_list[0]
            if result is None:
                result = result_list[1]
                

    return result



# def create_location_category_dict():
#     location_category_dict = {}
#     location_category_csv = pandas.read_csv(LOCATION_CATEGORY_FILE)
#     location_category_data_set = location_category_csv.iloc[ : , : ].values
#     for location_category_data in location_category_data_set:
#         key = str(location_category_data[0]) + '&' +str(location_category_data[1])
#         location_category_dict[key] = location_category_data[2]

#     return location_category_dict

# def create_weather_dict():
#     weather_dict = {}
#     weather_csv = pandas.read_csv(WEATHER_FILE)
#     weather_data_set = weather_csv.iloc[ : , : ].values
#     for weather_data in weather_data_set:
#         data_time_str = weather_data[0]
#         data_time = datetime.datetime.strptime(data_time_str, '%Y-%m-%d %H:%M:%S')
#         data_dict = {
#             'humidity' : weather_data[1],
#             'pressure' : weather_data[2],
#             'temperature' : weather_data[3],
#             'weather_description' : weather_data[4],
#             'wind_direction' : weather_data[5],
#             'wind_speed' : weather_data[6]
#         }
#         weather_dict[data_time] = data_dict
#     return weather_dict


# def test():
#     weather_csv = pandas.read_csv(WEATHER_FILE)
#     weather_data_set = weather_csv.iloc[ : , : ].values
#     a = numpy.where(weather_data_set = 41.903932)
#     print(a)

if __name__ == '__main__':
    # weather_dict = create_weather_dict()

    # location_category_dict = create_location_category_dict()

    # pprint.pprint(weather_dict)
    main()