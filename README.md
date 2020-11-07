# Knowledge Distillation

## Task Instructions
In the data directory, there is fin_news_all-data.csv containing 3-class 
sentiment training data with news headlines and corresponding sentiment. In 
represents investors sentiment, i.e. for example, if the news article might 
lead to the rise of the company shares price it is marked as positive.  

1. Create a simple NN for classification using glove words embedding, embedding 
layer and several CNN (preferably) or GRU/LSTM layers. Getting best possible 
accuracy is not the point of this step.

2. notebooks folder contains a quick start inference example of RoBERTa 
model trained on the same dataset using fast-bert. You can download model at

https://sagemaker-data-nlp.s3.amazonaws.com/fin-news-sentiment-multilabel/output/fin-news-sentiment-multilabel-2020-08-26-13-07-53-550/output/model.tar.gz 

You need to distil knowledge from this model to the model you created at step 
1. Knowledge distillation is a technique where you train a smaller model to 
mimic the output of the bigger one. Here we want to distil softmax outputs of 
the bigger model. You can use data/headlines.csv for distillation however 
experimenting with a much bigger corpus of arbitrary sentences might be 
beneficial as well. Please provide a script which will take as an input model in 
fast-bert format and a CSV with data for distillation and will create a 
distilled model. Compare the performance of the distilled model with the performance
of the model trained from scratch.

If you have any questions about the task or need some clarification please 
ask alex@permutable.ai
