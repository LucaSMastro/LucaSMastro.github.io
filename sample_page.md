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

### 2. Formating Locations

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
