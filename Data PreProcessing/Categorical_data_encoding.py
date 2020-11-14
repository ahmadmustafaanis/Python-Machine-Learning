import os
import pandas as pd
import numpy as np

df = pd.DataFrame([
['green', 'M', 10.1, 'class2'],
['red','L','13.5','class1'],
['blue','XL',15.3,'class2']
], columns=['color','size','price','classlabel'])


def cls():
	print("Press Enter to continue")
	input()
	os.system('clear')

print("Orinal Data frame is ", df, sep='\n')

cls()

#let's map the ordinal features. i.e Xl=3, L=2, M=1

size_mapping ={'XL':3,
                'L':2,
                'M':1
              }

df['size']= df['size'].map(size_mapping)

print("Data Frame after Mapping is ", df, sep='\n')
cls()
#reverse mapping
inv_size_mapping = {v:k for k,v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print("Reverse Mapping Df",df,sep='\n')

#encoding the label variable.
cls()

print("Orignal Data Frame is ", df, sep='\n')

class_mapping = {label: idx for idx,label in enumerate(np.unique(df['classlabel']))}

df['classlabel'] = df['classlabel'].map(class_mapping)

print("Encoding the classlabel", df, sep='\n')
cls()
print("data frame is", df, sep='\n')
print("\n Reversing the DataFrame")
rev_class_mapping = {i:j for j,i in class_mapping.items()}

df['classlabel']= df['classlabel'].map(rev_class_mapping)

print("Data Frame after Reversing is ", df, sep='\n')
cls()
# Label Encoding via SkLearn

from sklearn.preprocessing import LabelEncoder
print("Data Frame is", df, sep='\n')
le = LabelEncoder()

df['classlabel'] = le.fit_transform(df['classlabel'].values)

print("Data Frame after sklearn label encoding is ", df, sep='\n')

cls()
print("Data Frame right now", df, sep='\n')
df['classlabel'] = le.inverse_transform(df['classlabel'])
print("Using ._inverse_transform on df", df)

cls()
