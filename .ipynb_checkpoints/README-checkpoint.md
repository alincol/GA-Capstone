# Predicting Climbing Grade by Description
It is generally known in the climbing community that grades are subjective. The Yosemite Decimal System, or YDS grade, is commonly used throughout the United States. Yet this system, which is meant to be objective, can have different nuance from climbing area to climbing area. My initial goal was to use the description to classify routes as 'sandbagged' or not, indicating that the given grade of the route is lower than the route's true difficulty. To explore that problem, however, I first need to show that description can be correlated to climbing grade.

## Problem Statement
Create a classification model with an accuracy over the baseline that can be used to predict the grade of a route based on the description.

## Dataset
For this initial exploration I chose the most recent scrape for the US from [OpenBeta](https://openbeta.io/), a project that aims to make climbing beta open source and easily accessible. The project was created after [Mountain Project](https://www.mountainproject.com/), the most popular website for crowdsourced climbing beta, was sold to an owner much less keen on sharing the data they had collected. The dataset I used, from [this OpenBeta branch](https://github.com/OpenBeta/climbing-data/tree/next), contained 127,000 roped climbing routes that had been recently scraped from Mountain Project. From the given features, I created the following for modeling:

| Feature                  | Type | Description                                                                                   |
|--------------------------|------|-----------------------------------------------------------------------------------------------|
| grade_reduced            | int  | The plain difficulty rating of the YDS grade (only 5th class routes were included)            |
| lemmatized_text_combined | str  | The combination of the three main text features, with various cleaned and lemmatized versions |
| type                     | bit  | Actually 7 columns, each a one-hot representation of the climb type of the route              |
| year_established         | int  | The year the route was established, extracted from the description of the FA                  |

## Implementation
In exploring this problem, I tried three different modeling approaches. The main modeling approach was using Facebook's fastText supervised model, which is similar to word2vec with a hierarchical softmax output. For transfer learning, I tried a Bert pre-trained model loaded from HuggingFace's transformers. I also implemented Multinomial Naive Bayes from sklearn. Each of these models required slightly different pre-processing, but they were all run on the version of the combined text that had stop words removed and was lemmatized using spaCy. 

The primary challenge of this dataset is deep class imbalances. My reduction of the 5th class of the YDS system leads to 16 possible grades, 5.0-5.15. 

| Grade | Num Rows | % Dataset |
|-------|----------|-----------|
| 5.0   | 166      | 0.1       |
| 5.1   | 118      | 0.09      |
| 5.2   | 396      | 0.3       |
| 5.3   | 731      | 0.5       |
| 5.4   | 1,722    | 1.3       |
| 5.5   | 2,538    | 1.9       |
| 5.6   | 5,424    | 4.2       |
| 5.7   | 9,953    | 7.8       |
| 5.8   | 13,715   | 10.7      |
| 5.9   | 16,556   | 13.03     |
| 5.10  | 33,577   | 26.4      |
| 5.11  | 24,114   | 18.9      |
| 5.12  | 13,829   | 10.8      |
| 5.13  | 3,710    | 2.9       |
| 5.14  | 451      | 0.3       |
| 5.15  | 8        | 0.006     |

Baseline models achieved around 33% accuracy, and unfortunately, an array of extra data processing steps and model iterations yielded no better results. 

## Further Considerations