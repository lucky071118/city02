
MAPPING_LIST = [
	'snow',
	'rain & thunderstorm',
	'sunny & overcast',
	'smoke & foggy'
]




def get_weather_feature_size():
	return len(MAPPING_LIST)

def get_weather_description_category(weather_description):
	
	result = []

	if weather_description == 'broken clouds':
		result.append('sunny & overcast')
	elif weather_description == 'drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'dust':
		result.append('smoke & foggy')
	elif weather_description == 'few clouds':
		result.append('sunny & overcast')
	elif weather_description == 'fog':
		result.append('smoke & foggy')
	elif weather_description == 'haze':
		result.append('smoke & foggy')
	elif weather_description == 'heavy intensity drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'heavy intensity rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'heavy snow':
		result.append('snow')
	elif weather_description == 'light intensity drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'light rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'light snow':
		result.append('snow')
	elif weather_description == 'mist':
		result.append('smoke & foggy')
	elif weather_description == 'moderate rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'overcast clouds':
		result.append('sunny & overcast')
	elif weather_description == 'proximity shower rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'proximity thunderstorm':
		result.append('rain & thunderstorm')
	elif weather_description == 'proximity thunderstorm with drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'proximity thunderstorm with rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'scattered clouds':
		result.append('sunny & overcast')
	elif weather_description == 'sky is clear':
		result.append('sunny & overcast')
	elif weather_description == 'smoke':
		result.append('smoke & foggy')
	elif weather_description == 'snow':
		result.append('snow')
	elif weather_description == 'thunderstorm':
		result.append('rain & thunderstorm')
	elif weather_description == 'thunderstorm with drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'thunderstorm with heavy rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'thunderstorm with light drizzle':
		result.append('rain & thunderstorm')
	elif weather_description == 'thunderstorm with light rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'thunderstorm with rain':
		result.append('rain & thunderstorm')
	elif weather_description == 'very heavy rain':
		result.append('rain & thunderstorm')

	return result



		
def encode_weather_category_list(weather_category_list):
	weather_category_encoding_list = []

	for category_name in MAPPING_LIST:
		if category_name in weather_category_list:
			weather_category_encoding_list.append(1)
		else:
			weather_category_encoding_list.append(0)
	return weather_category_encoding_list

if __name__ == '__main__':
	weather_category_list = get_weather_description_category('dust')
	weather_category_encoding_list = encode_weather_category_list(weather_category_list)
	print(weather_category_encoding_list)


