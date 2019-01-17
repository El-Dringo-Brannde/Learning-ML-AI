import pandas
import webbrowser
import os

data_table = pandas.read_csv('ml_house_data_set.csv')
html = data_table[0:100].to_html()
