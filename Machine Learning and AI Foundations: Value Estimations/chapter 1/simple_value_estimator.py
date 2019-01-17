def estimate_home_value(size_in_sqft, bedroom_number):
    # base home value
    value = 50000

    # add in the variable of sqft
    value = value + (size_in_sqft * 92)

    # adjust for number of bedrooms
    value = value + (bedroom_number * 10000)

    return value


valued = estimate_home_value(3800, 5)


print(f'Estimate Value: {valued}')
