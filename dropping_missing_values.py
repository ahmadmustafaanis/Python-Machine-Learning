import os
import pandas as pd 
from io import StringIO

def cls():
	print("Press Enter to continue")
	input()
	os.system('clear')

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''


df = pd.read_csv(StringIO(csv_data))

print(f"df before imputing missing values is\n{df}")
cls()
print(f"Amount of Missing Values in df is", df.isnull().sum(), sep='\n')
cls()
print("Df before||\n",df)

#using dropna

print(f"Data Frame using df.dropna(axis=0) which will drop all null rows is\n{df.dropna(axis=0)}") 

print(f"Data Frame using df.dropna(axis=1) which will drop all null columns is\n{df.dropna(axis=1)}") 

cls()

print(df)
print("Dropping all the rows where all the columns are null(i.e all values of row is null", df.dropna(how='all'), sep='\n')

print("Dropping all the rows that have less then 4 null values", df.dropna(thresh=4), sep='\n')

cls()

print("Orignal Data Frame", df, sep='\n')

print("Only dropping those rows where we have null in column C", df.dropna(subset=['C']), sep='\n')
