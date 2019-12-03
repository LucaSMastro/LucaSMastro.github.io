**Predicting Empty Seats on Flights:** As a frequent traveler, I often find myself hoping for flights to be mostly empty. This is, of course, difficult to predict given that airlines overfill flights based on the same predictors that would otherwise be useful in finding the number of empty seats. However, generating such a model to predict empty seats is useful for more than just curiosity. Value would be generated for any travel company capable of differentiating themselves from their competitors by recommending emptier flights.

### 1. Loading the data

I opted to use a dataset from the US Census Bureau containing records of domestic flights from 1990-2009

```python
import pandas as pd
import numpy as np
import re

#Read in flight information, adjust column titles.
flightdf = pd.read_csv('C:/datasets/flight_edges.csv', delimiter='\t')
flightdf.columns = ['Origin','Destination','Origin_City','Destination_City','Passengers','Seats','Flights',
                   'Distance','Fly_Date','Origin_Population','Destination_Population']
print(flightdf.head())

#Create the following columns:
#Empty_Seats is how many seats were empty
#Percent_Empty is the percent of seats that are empty
#low_passengers is 1 if more than 35% of the seats were empty or 0 otherwise.
flightdf['Empty_Seats'] = flightdf['Seats']-flightdf['Passengers']
flightdf['Percent_Empty'] = flightdf['Empty_Seats']/flightdf['Seats']
flightdf['Low_Passengers'] = np.where(flightdf['Percent_Empty']>0.35,1,0)
flightdf['Low_Passengers'].value_counts()
```
Which gives the following output:

<img src="images/LowPassengers1.PNG?raw=true"/>

### 2. Formating Locations

One might notice from the above output that locations are described by distinct airport codes. Given that over 500 airports in the US serve commercial flights, we need to generalize this to reduce the number of categories. First this is carried out by State.

```python
#There are too many airports to fit into bins for GBT and RF. We need to create variables that the tree can work with
#which still convey information from Origin and Destination features.
#I propose both sorting by states and converting to number of total flights.

#For doing it by state, for Origin_City and Destination_City, get states from cities and then put them through dictionary.


stateDict = {'AL':1, 'AK':2, 'AZ':3, 'AR':4, 'CA':5, 'CO':6, 'CT':7, 'DE':8,
            'FL':9, 'GA':10, 'HI':11, 'ID':12, 'IL':13, 'IN':14, 'IA':15, 'KS':16,
            'KY':17, 'LA':18, 'ME':19, 'MD':20, 'MA':21, 'MI':22, 'MN':23, 'MS':24,
            'MO':25, 'MT':26, 'NE':27, 'NV':28, 'NH':29, 'NJ':30, 'NM':31, 'NY':32,
            'NC':33, 'ND':34, 'OH':35, 'OK':36, 'OR':37, 'PA':38, 'RI': 39, 'SC':40,
            'SD':41, 'TN':42, 'TX':43, 'UT':44, 'VT':45, 'VA':46, 'WA': 47, 'WV':48,
            'WI':49, 'WY':50, 'DC':51}

flightdf['Origin_State'] = flightdf['Origin_City'].str.slice(start = -2)
flightdf['Origin_State'].replace(stateDict, inplace = True)

flightdf['Destination_State'] = flightdf['Destination_City'].str.slice(start = -2)
flightdf['Destination_State'].replace(stateDict, inplace = True)
flightdf.head()
```

Then an additional category is made classifying airports by frequency.

```python
#Finds how many flights flew from each origin and makes this into a column of our dataframe.

Origin_freq = {}
for index, row in flightdf.iterrows():
    if row['Origin'] in Origin_freq.keys():
        Origin_freq[row['Origin']] = Origin_freq[row['Origin']] + row['Flights']
    else:
        Origin_freq[row['Origin']] = row['Flights']
flightdf['Origin_Frequency'] = flightdf['Origin'].replace(Origin_freq)

flightdf.head()

#Finds how many flights flew to each destination and makes this into a column of our dataframe.

Destination_freq = {}
for index, row in flightdf.iterrows():
    if row['Destination'] in Destination_freq.keys():
        Destination_freq[row['Destination']] = Destination_freq[row['Destination']] + row['Flights']
    else:
        Destination_freq[row['Destination']] = row['Flights']
flightdf['Destination_Frequency'] = flightdf['Destination'].replace(Destination_freq)

flightdf.head()
```
With these categories created, we can perform machine learning while also utilizing geographical information.

### 3. Training a model

We will use PySpark in order to build our model.

```python
#Finding PySpark
import findspark
findspark.init()

import pyspark
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

conf = (SparkConf().setAppName('Flights Low Passengers'))
#Acessing additional memory
conf.set('spark.driver.memory', '8g')
sc = pyspark.SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#Converting our dataframe from pandas
flightdfs = sqlContext.createDataFrame(flightdf)
```
We will use a random forest model owing to its ability to work well with categorical features while avoiding overfitting.

```python
#Pipeline: Vector Assembler -> RF
assembler = VectorAssembler(inputCols=['Origin_State','Destination_State','Origin_Frequency','Destination_Frequency','Fly_Date','Distance'],
                            outputCol='features')
rf = RandomForestClassifier(featuresCol=assembler.getOutputCol(), labelCol='Low_Passengers', maxBins=64)
pipelineRF = Pipeline(stages=[assembler, rf])

train, test = flightdfs.randomSplit([0.7, 0.3])

modelRF = pipelineRF.fit(train)
```

### 4. Evaluation

```python
predictionRF = modelRF.transform(test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Low_Passengers", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictionRF)
print("Test Error = %g" % (1.0 - accuracy))
```

### 5. Discussion
