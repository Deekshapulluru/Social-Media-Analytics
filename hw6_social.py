"""
Social Media Analytics Project
Name:
Roll Number:
"""

from pickle import POP
import hw6_social_tests as test 
from collections import Counter

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [" ","\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df=pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    f1=fromString.find(":")
    f2=fromString.find("(")
    #print (f1)
    #print (f2)
    str=fromString[f1+2:f2-1:]
    #print (str)
    return str


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    f1=fromString.find("(")
    f2=fromString.find("from")
    str=fromString[f1+1:f2-1:]
    return str 


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    f1=fromString.find("from")
    f2=fromString.find(")")
    str=fromString[f1+5:f2:]
    #print (str)
    return str 


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    tags=message.split("#")
    #print(tags)
    hashtags=[]
    temp_word=""
    for i in range(1,len(tags)):
        for tag in tags[i]:
            #print(tag)
            if tag in endChars:
                break 
            else:
                temp_word+=tag 
        temp_word="#"+temp_word
        hashtags.append(temp_word)
        temp_word=""
    return hashtags



'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    #stateDf=pd.read_csv("statemappings.csv")
    #print("stateDf")
    #print(stateDf["state"])
    str=stateDf.loc[stateDf["state"] == state,'region']
    #print(str.values[0])
    #print(type(str))
    return str.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    name_list=[]
    position_list=[]
    state_list=[]
    region_list=[]
    hashtags_list=[]
    #data["name"]=[]
    #data["position"]=[]
    #data["state"]=[]
    #data["region"]=[]
    #data["hashtags"]=[]
    for index,row in data.iterrows():
        row_label=row['label']
        #print(str)
        names=parseName(row_label)
        name_list.append(names)
        position=parsePosition(row_label)
        position_list.append(position)
        state=parseState(row_label)
        state_list.append(state)
        region=getRegionFromState(stateDf,state)
        region_list.append(region)
        text_label=row["text"]
        hashtags=findHashtags(text_label)
        hashtags_list.append(hashtags)
    data["name"]=name_list
    data["position"]=position_list
    #print(data["position"])
    data["state"]=state_list
    data["region"]=region_list
    data["hashtags"]=hashtags_list
    return None 


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if (score<-0.1):
        return "negative"
    elif (score>0.1):
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments_list=[]
    for index,row in data.iterrows():
        text_label=row['text']
        s=findSentiment(classifier,text_label)
        sentiments_list.append(s)
    data["sentiment"]=sentiments_list
    #print(data)
    return 


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    #print(data["sentiment"])
    #data=addSentimentColumn(data)
    #print(data.columns)
    dict1 ={} 
    for index, row in data.iterrows(): 
        if colName!="" and dataToCount!="":
            #print(colName) 
            if (data[colName][index] == dataToCount): 
                if (data["state"][index] in dict1): 
                    dict1[data["state"][index]] += 1 
                else: 
                    dict1[data["state"][index]] = 1 
        else: 
            if data["state"][index] in dict1: 
                dict1[data["state"][index]] += 1 
            else: 
                dict1[data["state"][index]] = 1
    return dict1



'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    dict={}
     #    # dict[key]={}
    for index, row in data.iterrows():  
        if row["region"] not in dict: 
            dict[row["region"]]={}
        if row[colName] in dict[row["region"]]:
            dict[row["region"]][row[colName]]+=1
        else:
            dict[row["region"]][row[colName]]=1
    #print(dict)
    return dict 


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtags_dict={}
    for index,row in data.iterrows():
        hash_tags=row["hashtags"]
        for i in range(len(hash_tags)):
            if hash_tags[i] not in hashtags_dict:
                hashtags_dict[hash_tags[i]]=1
            else:
                hashtags_dict[hash_tags[i]]+=1
    #print(hashtags_dict)
    return hashtags_dict 
'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    sort={}
    res={}
    #data=pd.read_csv("data/politicaldata.csv")
    #df1=pd.read_csv("data/statemappings.csv")
    #addColumns(data,df1)
    #hashtags=getHashtagRates(data)
    x=Counter(hashtags)
    sort=sorted(x.items(),key=lambda k: -k[1])
    #print(sort)
    i=0
    while (i<count):
        #res[key][i]=sort[i][1]
        res[sort[i][0]]=sort[i][1]
        i=i+1
    return res




'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    dict2 = [] 
    for index,row in data.iterrows(): 
        col = row['text'] 
        if hashtag in col: 
            sent_value = row["sentiment"] 
            if sent_value == "positive": 
                dict2.append(1) 
            if sent_value == "negative": 
                dict2.append(-1) 
            if sent_value == "neutral": 
                dict2.append(0) 
    avg = sum(dict2)/len(dict2) 
    return avg
    
    # for index,row in data.iterrows():
    #     col=row["hashtags"]
    #     #print(col)
    #     for i in range(len(col)):
    #         print (col[i])

### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    """print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()"""

    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""
    df=pd.read_csv("data/politicaldata.csv")
    df1=pd.read_csv("data/statemappings.csv")
    #test.testMakeDataFrame()
    #test.testParseName()
    #test.testParsePosition()
    #est.testParseState()
    #est.testFindHashtags()
    #est.testGetRegionFromState()
    #test.testAddColumns()
    #test.testFindSentiment()
    #test.testAddSentimentColumn()
    addSentimentColumn(df)
    addColumns(df,df1)
    test.testGetDataCountByState(df)
    test.testGetDataForRegion(df)
    test.testGetHashtagRates(df)
    test.testMostCommonHashtags(df)
    test.testGetHashtagSentiment(df)
