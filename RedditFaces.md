# Reddit Image Data Collection
The following script can be utilized to scrape a dataset of submission images with faces, titles and comments from a user-provided subreddit. As an example, run against an attractiveness rating subreddit such as r/rateme, these comments could be used to extrapolate crowd-sourced labels of facial attractiveness. The title of each post is also collected, which in the aforementioned use case could be utilized to infer gender and age as further features for analysis.

Attention is paid while scraping to curate results to remove missing links or irrelevant images (as may result from either mischeivous posters or removed images whose links have been recycled.) However, auditing the dataset for unexpected images owing to the aforementioned causes may identify a need for further data cleaning.

```python
import requests
import re
import os
import pandas as pd
import time
import cv2

#Comments and Posts may contain emojis. We will remove them
def omitEmojis(s):
    return(''.join([c for c in s if ord(c) < 65535]))

#This script will make many API calls. To make the script more robust to connection problems, we will create a function to retry these if a TimeoutError is raised.
def handleConnectionErrors(request, validStatus):
    i=0
    while i<10:
        i+=1
        try:
            response=requests.get(request)
        except Exception as e:
            print('ERROR RAISED IN REQUEST:',e)
            print('RETRYING',i,'OF 10 IN 15 SECONDS.')
            time.sleep(15)
            continue
        if response.status_code not in validStatus:
            print('REQUEST STATUS CODE DOES NOT INDICATE SUCCESS. CODE:',response.status_code)
            print('RETRYING',i,'OF 10 IN 15 SECONDS.')
            time.sleep(15)
            continue
        return response
    return requests.get(request)
    

submissionList = list()
removedList = list()

size=500
iterations=80
subreddit='truerateme'

usefulIndex=0
finalIndex=0
mainLoopIndex = 0
lastCreated=0

#making an API call to pushshift.io and grabbing json data on 'size' submissions from 'subreddit'.
response=handleConnectionErrors('https://api.pushshift.io/reddit/submission/search/?subreddit='+subreddit+'&num_comments=>1&size='+str(size), [200])

print('Beginning scrape of subreddit:',subreddit,'over',iterations,'iterations of',size,'submissions.')

#The above API has a limit of 500 returned submissions (of which far fewer than 500 will actually be useful (having comments, and valid images)). To gain more data, we will make repeated calls through a while loop.
while (mainLoopIndex<iterations):

    if(response.json()):
        for post in response.json()['data']:
            try:
                imageURL = list()
                #grabbing Image URL
                #handling posts with many images
                if 'gallery_data' in post.keys():
                    #grabbing each image's ID, the appropriate file ending from the metadata, and coercing into a URL.
                    for media in post['gallery_data']['items']:
                        mediaID = media['media_id']
                        if post['media_metadata'][mediaID]['status'] == 'valid':
                            fileEnding = post['media_metadata'][mediaID]['m'].split('/')[1]
                            imageURL.append(omitEmojis('https://i.redd.it/'+mediaID+'.'+fileEnding))
                    
                elif 'url' in post.keys():
                    #posts with a single image will have that image returned under the url key.
                    if re.search("\.jpg$", post['url']) or re.search("\.png$", post['url']) or re.search("\.gif$", post['url']):
                        imageURL.append(omitEmojis(post['url']))

                #If an image was found and captured, also collect postID, datecreated, and title
                if imageURL:
                    submissionID = post['id']
                    postLink = post['full_link']
                    dateCreated = post['created_utc']
                    submissionTitle = post['title']
                    #find the comments associated with that postID
                    commentResponse = requests.get('https://api.pushshift.io/reddit/comment/search/?link_id='+submissionID)
                    commentsList=list()
                    for comment in commentResponse.json()['data']:
                        if comment['body'] not in ('[removed]','[deleted]'):
                            commentsList.append(comment['body'])
                    #if comments were found, that were neither removed nor deleted, then proceed
                    if commentsList:
                        for link in imageURL:
                            submissionList.append([link,submissionID,dateCreated,submissionTitle,commentsList])
                            usefulIndex+=1
            except ValueError:
                    print('JSON DECODE ERROR IN PARSING POST. SKIPPING.')

    #Submit a new submission request to retrieve submissions that are dated before the oldest submission we currently have.
    
    response=handleConnectionErrors('https://api.pushshift.io/reddit/submission/search/?subreddit=truerateme&num_comments=>1&size='+str(size)+'&before='+str(dateCreated), [200])

    mainLoopIndex+=1
    print('PROGRESS UPDATE: Have scraped',str(mainLoopIndex*size),'submissions of which',usefulIndex,'images with comments were found.')
    
#From the output of the above while loop, we need to parse the list as follows:
#Download the images to unique files in folders whose name is in accordance with the submissionID. Names for these images can be found by splitting url on '/' delimiter and taking final element.
#Create a dataframe out of all other elements of the list.

print('PROGRESS UPDATE: Scrape completed. Saving.')

#Declaring a cascade classifier using the OpenCV model
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Retrieving the files from the scraped URLs and saving them, constructing a DataFrame and saving it.
for submission in submissionList:
    #Downloading submission image to an interim location
    response = handleConnectionErrors(submission[0],[200,404])
    imgExt = '.'+submission[0].split('.')[-1]
    interimPath='interim'+imgExt
    file=open(interimPath,'wb')
    file.write(response.content)
    file.close()
    
    #If image is identical to a 'file not found' image used by Reddit, skip it and add it to a list to be removed from submissions.
    if open(interimPath,'rb').read() == open('imageRemoved'+imgExt,'rb').read():
        removedList.append(submission)
        os.remove(interimPath)
        continue

    #Images may have been replaced, or the URL of deleted images may have been reused. To reduce occurrence of this problem, we will keep only images where a face is detected.
    #To run this for collection of images that feature something other than faces, some other detection model may be useful for reducing occurrence of random images.
    try:
        interimImg=cv2.imread(interimPath)
        grayImg=cv2.cvtColor(interimImg, cv2.COLOR_BGR2GRAY)
        if len(face_cascade.detectMultiScale(grayImg, 1.05, 3))==0:
            removedList.append(submission)
            os.remove(interimPath)
            print('PROGRESS UPDATE: Image Removed: Face not found')
            continue
    except Exception as e:
        print('ERROR IN RUNNING FACE RECOGNITION ON IMAGE:',e)
        continue

    finalPath='data/'+submission[1]+'/'
    #If path to submissionID does not exist, create it.
    if not os.path.exists(finalPath):
        os.makedirs(finalPath)
    #move file from interim location to submissionID/photoID
    os.replace(interimPath,finalPath+submission[0].split('/')[-1])
    finalIndex+=1
    #reformat from URL to photoID for use in DataFrame
    submission[0]=submission[0].split('/')[-1]

for removed in removedList:
    #Remove entries corresponding to missing images from SubmissionsList
    submissionList.remove(removed)

#Construct and save a DataFrame
redditAttractDF=pd.DataFrame(submissionList, columns=['ImageID','SubmissionID','DateCreated','SubmissionTitle','Comments'])
redditAttractDF.to_csv('data/redditAttractiveness.csv', encoding='utf-16',index=False)

print('PROGRESS UPDATE: Save Complete.',finalIndex,'saved images.')
```
