import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import  classification_report,confusion_matrix

# Read dataset csv with ; seperator
df = pd.read_csv ('clinical_dataset.csv',sep=';')

# Preprocessing of the clinical dataset
# Convert nominal features to numerical
df['fried'].replace(['Non frail','Pre-frail','Frail'],[0,1,2],inplace=True)
df['gender'].replace(['F','M'],[0,1],inplace=True)
df['ortho_hypotension'].replace(['No','Yes'],[0,1],inplace=True)
df['vision'].replace(['Sees poorly','Sees moderately','Sees well'],[0,1,2],inplace=True)
df['audition'].replace(['Hears poorly','Hears moderately','Hears well'],[0,1,2],inplace=True)
df['weight_loss'].replace(['No','Yes'],[0,1],inplace=True)
df['balance_single'].replace(['<5 sec','>5 sec'],[0,1],inplace=True)
df['gait_optional_binary'] = df['gait_optional_binary'].astype(int)
df['gait_speed_slower'].replace(['No','Yes'],[0,1],inplace=True)
df['grip_strength_abnormal'].replace(['No','Yes'],[0,1],inplace=True)
df['low_physical_activity'].replace(['No','Yes'],[0,1],inplace=True)
df['memory_complain'].replace(['No','Yes'],[0,1],inplace=True)
df['sleep'].replace(['No sleep problem','Occasional sleep problem','Permanent sleep problem'],[0,1,2],inplace=True)
df['living_alone'].replace(['No','Yes'],[0,1],inplace=True)
df['leisure_club'].replace(['No','Yes'],[0,1],inplace=True)
df['house_suitable_participant'].replace(['No','Yes'],[0,1],inplace=True)
df['house_suitable_professional'].replace(['No','Yes'],[0,1],inplace=True)
df['health_rate'].replace(['1 - Very bad','2 - Bad','3 - Medium','4 - Good','5 - Excellent'],[0,1,2,3,4],inplace=True)
df['health_rate_comparison'].replace(['1 - A lot worse','2 - A little worse','3 - About the same','4 - A little better','5 - A lot better'],[0,1,2,3,4],inplace=True)
df['activity_regular'].replace(['No','< 2 h per week','> 2 h and < 5 h per week','> 5 h per week'],[0,1,2,3],inplace=True)
df['smoking'].replace(['Never smoked','Past smoker (stopped at least 6 months)','Current smoker'],[0,1,2],inplace=True)

# Remove erroneous values
missing_values=[999,'test non realizable','Test not adequate']
df.replace(missing_values,'',inplace=True)
#df.to_csv('bigdataclinical.csv',sep=';')

# Handle missing values
# Replace empty values with NaN object
df.replace('', np.nan, inplace=True)
# Remove entries with missing values in some features
df.dropna(inplace=True)

# Classification with KNN
# Drop fried and 5 parameters used for generating the fried categorization
X = df.drop(columns=['fried','weight_loss','exhaustion_score','gait_speed_slower','grip_strength_abnormal','low_physical_activity'])

# Assign fried column to y
y = df['fried'].values

# Split dataset to train and test sets with 20% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
start = time.time()
# Initialize KNN classifier with number of neighbors to 5
knn = KNeighborsClassifier(n_neighbors = 5)
# Fit the k-nearest neighbors classifier from the training dataset
knn.fit(X_train,y_train)
# Make the prediction
y_pred = knn.predict(X_test)
# Model Accuracy
accuracy = float("{:.2f}".format(metrics.accuracy_score(y_test, y_pred) * 100))
print("KNN Accuracy:",accuracy, "%")
end = time.time()
print(end - start, "seconds")
start = time.time()
#classsification with random forest
clf=RandomForestClassifier(n_estimators=25,random_state=1)
#Fit the random forest classifier from the training dataset
clf.fit(X_train,y_train)
# Make the prediction
y_pred=clf.predict(X_test)
# Model Accuracy
accuracy = float("{:.2f}".format(metrics.accuracy_score(y_test, y_pred) * 100))
#print(classification_report(y_test, y_pred))
print("RF Accuracy:",accuracy, "%")
end = time.time()
print(end - start, "seconds")

#print(confusion_matrix(y_test, y_pred))