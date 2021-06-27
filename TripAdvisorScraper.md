# TripAdvisor Reviews Scraping
The following script enables scraping a user-specified quantity of the most recent reviews for a specified attraction. This DataFrame can then be utilized in other projects. (Here, it is utilized to analyze Walt Disney World Reviews.)


```python
from bs4 import BeautifulSoup
import requests
from requests import get
import pandas as pd
```


```python
#URL of front page of TripAdvisor attraction (here is for Walt Disney World)
url = 'https://www.tripadvisor.com/Attraction_Review-g34515-d143394-Reviews-Walt_Disney_World_Resort-Orlando_Florida.html'

#Setting index of reviews scraped to 0
i=0
#number of reviews to scrape (will round up to nearest ten)
numberReviews = 5000
#This list will store reviews prior to conversion to a pandas dataframe 
reviewsList = list()
```


```python
#Creating a function for cleaning undesirable characters that are left when converting from BeautifulSoup to string
def cleanHTMLRelics(string):
    for char in ['[','\'','\"',']','<br/>']:
        string = string.replace(char,'')
    return string
```


```python
#Each iteration of this loop scrapes a page, then goes to the next page.
while(i<numberReviews):
    if(i%100==0):
        print('Beginning scrape of review: '+str(i))
    #Pulling from URL
    results = requests.get(url)

    soup = BeautifulSoup(results.text, 'html.parser')
    #Finding class containing Reviews
    soup = soup.find(attrs={"class":"_1c8_1ITO"})
    soup = soup.contents[:-1]

    #For each review, scrape the rating, title, contents and date, then assemble these into a list.
    for review in soup:
        
        #scraping rating
        ratingScore = review.find(attrs={"class":"zWXXYhVR"})
        ratingScore = str(ratingScore['aria-label']).split()[0]
        ratingScore = float(ratingScore)
        
        #scraping date
        reviewDate = review.find(attrs={"class":"DrjyGw-P _26S7gyB4 _1z-B2F-n _1dimhEoy"})
        reviewDate = str(reviewDate.contents)
        reviewDate = ' '.join(reviewDate.split()[1:])
        reviewDate = cleanHTMLRelics(reviewDate)
        
        #scraping review title and contents
        titleContents = review.find_all(attrs={"class":"_2tsgCuqy"})
        title=str(titleContents[0].contents)
        title=cleanHTMLRelics(title)
        
        contents = str(titleContents[1].contents)
        contents = cleanHTMLRelics(contents)
        
        reviewAsList = [ratingScore, reviewDate, title, contents]
        reviewsList.append(reviewAsList)

    i+=10
    #Change url to the page with next ten reviews.
    url = 'https://www.tripadvisor.com/Attraction_Review-g34515-d143394-Reviews-or'+str(i)+'-Walt_Disney_World_Resort-Orlando_Florida.html'
    
```

    Beginning scrape of review: 0
    Beginning scrape of review: 100
    Beginning scrape of review: 200
    Beginning scrape of review: 300
    Beginning scrape of review: 400
    Beginning scrape of review: 500
    Beginning scrape of review: 600
    Beginning scrape of review: 700
    Beginning scrape of review: 800
    Beginning scrape of review: 900
    Beginning scrape of review: 1000
    Beginning scrape of review: 1100
    Beginning scrape of review: 1200
    Beginning scrape of review: 1300
    Beginning scrape of review: 1400
    Beginning scrape of review: 1500
    Beginning scrape of review: 1600
    Beginning scrape of review: 1700
    Beginning scrape of review: 1800
    Beginning scrape of review: 1900
    Beginning scrape of review: 2000
    Beginning scrape of review: 2100
    Beginning scrape of review: 2200
    Beginning scrape of review: 2300
    Beginning scrape of review: 2400
    Beginning scrape of review: 2500
    Beginning scrape of review: 2600
    Beginning scrape of review: 2700
    Beginning scrape of review: 2800
    Beginning scrape of review: 2900
    Beginning scrape of review: 3000
    Beginning scrape of review: 3100
    Beginning scrape of review: 3200
    Beginning scrape of review: 3300
    Beginning scrape of review: 3400
    Beginning scrape of review: 3500
    Beginning scrape of review: 3600
    Beginning scrape of review: 3700
    Beginning scrape of review: 3800
    Beginning scrape of review: 3900
    Beginning scrape of review: 4000
    Beginning scrape of review: 4100
    Beginning scrape of review: 4200
    Beginning scrape of review: 4300
    Beginning scrape of review: 4400
    Beginning scrape of review: 4500
    Beginning scrape of review: 4600
    Beginning scrape of review: 4700
    Beginning scrape of review: 4800
    Beginning scrape of review: 4900
    


```python
#Converting reviewsList into a pandas dataframe
reviewsDF = pd.DataFrame(reviewsList, columns = ['Rating','Date','Title','Contents'])
#Saving dataframe to a csv
reviewsDF.to_csv('TripAdvisorReviews.csv')
```


```python

```
