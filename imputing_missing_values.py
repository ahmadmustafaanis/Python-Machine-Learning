import os
import pandas as pd
from io import StringIO
#imputing via sklearn

from sklearn.impute import SimpleImputer
import numpy as np

def cls():
	print("Press Enter to continue")
	input()
	os.system('clear')

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
	12.0,12.0,12.0,12.0'''

df = pd.read_csv(StringIO(csv_data))


print("Orignal Data Frame is ", df, sep='\n')

#creating SimpleImputer for imputing

imr = SimpleImputer(missing_values=np.nan, strategy='mean')

imr.fit(df.values)
imputed_data = imr.transform(df.values)
print("Replaced each nan with mean value of that column", imputed_data, sep='\n')
cls()

print("Orignal Data Frame", df, sep='\n')

new_imr = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

new_imr.fit(df.values)

imputed_data = new_imr.transform(df.values)
print("Replaced each value with most frequent value", imputed_data, sep='\n')
