import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
table=pd.read_csv("table.csv")
x=LabelEncoder()
table['Property_Area']=x.fit_transform(table['Property_Area'])
table['Dependents'] = x.fit_transform(table['Dependents'])
i=SimpleImputer(missing_values=np.nan, strategy='mean')
table[['LoanAmount', 'Loan_Amount_Term','Credit_History']]=i.fit_transform(table[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
x_new = pd.get_dummies(data=table,columns=['Gender', 'Married', 'Education', 'Loan_Status'],dtype=int)
x_new.to_csv("table2.csv",index=False)
print(x_new)


dataset = x_new.drop(columns=["Loan_ID","Self_Employed"])
normal=MinMaxScaler()
normalized=normal.fit_transform(dataset)
print(normalized)

y=x_new["Loan_Status_N"]
XNewOne= x_new.drop(columns=["Loan_Status_Y"])

X_train, X_test, y_train, y_test = train_test_split(XNewOne, y, test_size=0.30,random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
