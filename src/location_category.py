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