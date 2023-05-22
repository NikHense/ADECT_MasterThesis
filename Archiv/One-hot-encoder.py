# %%
# importing one hot encoder 
from sklearn.preprocessing import OneHotEncoder

# creating one hot encoder object 
onehotencoder = OneHotEncoder()

# select the columns to be one-hot encoded
columns_to_encode = ['Payment_Number', 'Object_Number',
                     'Country_Region_Code', 'Payment_Method_Code',
                     'Customer_IBAN', 'Vendor_IBAN_BIC',
                     'Vendor_Bank_Origin', 
                     'Created_By', 'Mandant']

# apply one-hot encoding to the selected columns
for col in columns_to_encode:
    data = total_payments[[col]]
    X = onehotencoder.fit_transform(data.values.reshape(-1,1)).toarray()
    dfOneHot = pd.DataFrame(X, columns=[col+"_"+str(int(i)) for i in range(X.shape[1])])
    total_payments = pd.concat([total_payments, dfOneHot], axis=1)
    total_payments.drop([col], axis=1, inplace=True)

# print the resulting dataframe
total_payments.info()

# %%
