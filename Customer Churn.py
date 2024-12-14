#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df


# In[126]:


df.info()


# In[127]:


df.isna()


# In[128]:


df.dropna()


# In[129]:


df


# In[130]:


df.describe()


# In[131]:


df.columns


# In[132]:


df


# In[133]:


#What is the distribution of churn across gender?

gender_churn = df.groupby(['gender', 'Churn']).size().unstack()

import matplotlib.pyplot as plt
gender_churn.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title("Distribution of Churn Across Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(["No Churn", "Churn"], title="Churn")
plt.tight_layout()
plt.show()


# In[134]:


#How does being a Senior Citizen affect churn rates?
senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()


fig, ax = plt.subplots(figsize=(8, 6))
senior_churn.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title("Churn Rates for Senior Citizens vs Non-Senior Citizens")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)  
plt.legend(["No Churn", "Churn"], title="Churn")
plt.tight_layout()
plt.show()


# In[135]:


#Does having InternetService (DSL, Fiber optic, or None) impact churn?

internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
internet_churn.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title("Churn Rates on InternetService")
plt.xlabel("internet service type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)  
plt.legend(["No Churn", "Churn"], title="Churn")
plt.tight_layout()
plt.show()



# In[ ]:





# In[136]:


#Do customers with multiple services like StreamingTV or StreamingMovies churn more?


def plot_churn_by_service(service_column, title):
    service_churn = df.groupby([service_column, 'Churn']).size().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    service_churn.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black')

    plt.title(title)
    plt.xlabel(f"{service_column}")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)  
    plt.legend(["No Churn", "Churn"], title="Churn")
    plt.tight_layout()
    plt.show()

plot_churn_by_service('StreamingTV', "Impact of StreamingTV on Churn")
plot_churn_by_service('StreamingMovies', "Impact of StreamingMovies on Churn")


# In[137]:


#What is the relationship between Contract type and churn?

contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
contract_churn.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title("Churn Rates on Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)  
plt.legend(["No Churn", "Churn"], title="Churn")
plt.tight_layout()
plt.show()


# In[138]:


#How does tenure (number of months as a customer) relate to churn?

churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(churned['tenure'], bins=20, color='#ff7f0e', alpha=0.7, label='Churned')
ax.hist(not_churned['tenure'], bins=20, color='#1f77b4', alpha=0.7, label='Not Churned')
plt.title("Distribution of Tenure for Churned vs Non-Churned Customers")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.show()


# In[139]:


#Do higher MonthlyCharges lead to higher churn?

plt.hist(df[df['Churn'] == 'Yes']['MonthlyCharges'], bins=20, color='#ff7f0e', alpha=0.7, label='Churned')
plt.hist(df[df['Churn'] == 'No']['MonthlyCharges'], bins=20, color='#1f77b4', alpha=0.7, label='Not Churned')

plt.title("Monthly Charges Distribution by Churn Status")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.show()


# In[140]:


#How does PaymentMethod affect churn?

payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
payment_churn.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.title("Churn Rates on Payment Method")
plt.xlabel("Payment Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)  
plt.legend(["No Churn", "Churn"], title="Churn")
plt.tight_layout()
plt.show()


# In[141]:


df


# In[150]:


df


# In[158]:


#Use a heatmap to show correlations between numerical features like tenure, MonthlyCharges, and TotalCharges.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()


# In[ ]:





# In[163]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn_Numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




