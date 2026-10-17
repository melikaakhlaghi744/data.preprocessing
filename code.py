import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
table=pd.read_csv("table.csv")
x=LabelEncoder()
table['Property_Area']=x.fit_transform(table['Property_Area'])
table['Dependents'] = x.fit_transform(table['Dependents'])
i=SimpleImputer(missing_values=np.nan, strategy='mean')
table[['LoanAmount', 'Loan_Amount_Term','Credit_History']]=i.fit_transform(table[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
x_new = pd.get_dummies(data=table,columns=['Gender', 'Married', 'Education', 'Loan_Status'],dtype=int)
x_new.to_csv("table2.csv",index=False)
print(x_new)
