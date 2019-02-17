from flask import Flask, render_template,url_for,request
from flask_mail import Mail, Message
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import pickle
from newsapi.articles import Articles
import newsapi
import sys
import speech_recognition as sr
import tweepy
import json
import smtplib
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
app = Flask(__name__)



def remove_new_lines(string):
	string = string.replace("\n","")
	return string
def remove_stop_words(string):
	example = string
	stop_words = set(stopwords.words('english')) 
	word_tokens = word_tokenize(example)
	filtered_sentence = [] 
	for w in word_tokens: 
		if w not in stop_words: 
			filtered_sentence.append(w)
	return ' '.join(filtered_sentence)
def strip(string):
	string = string.strip()
	return string
def remove_weird(string):
	string = string.replace("``","")
	string = string.replace('""',"")
	string = string.replace("''","")
	string = string.replace("  "," ")
	return string 

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/sendmail',methods=['POST'])
def sendmail():
	if request.method == 'POST':
		msg = request.form['report']
		server = smtplib.SMTP('smtp.gmail.com', 587)
		server.starttls()
		server.login("ayushkalla2050@gmail.com", "Hellofriends@123")
		server.sendmail("ayushkalla2050@gmail.com", "ayushkalla1996@gmail.com", msg)
		server.quit()
	return render_template('sendmail.html')
@app.route('/complain')
def complain():
	return render_template('complain.html')
@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')
@app.route('/twitter')
def twitter():
	consumer_key = "FK3UDBzcqWM4T6EReegI6ts2Z"
	consumer_secret = "CDthUQOA4RM3J2WOjLYKSjQ3fQbi2LisVjrt7hWfnzNo56DOAY"
	access_token = "1036391783528325121-iZykhcjhBKjfSYT1zcGAYbOfnfYx28"
	access_token_secret = "MpZak0zpmwWotKKb9C9ggg66VA0xjrgnicQfvIi4mNOIL"


	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	# Setting your access token and secret
	auth.set_access_token(access_token, access_token_secret)
	# Creating the API object while passing in auth information
	api = tweepy.API(auth) 

	# The Twitter user who we want to get tweets from
	name = "nytimes"
	# Number of tweets to pull
	tweetCount = 5

	# Calling the user_timeline function with our parameters
	results = api.user_timeline(id=name, count=tweetCount)

	# foreach through all tweets pulled
	tweets = []
	for tweet in results:
	   # printing the text stored inside the tweet object
	   tweets.append(tweet.text)
	refined_tweets = []
	for item in tweets:
	    try:
	        refined_tweets.append(item.split("https:")[0])
	    except:
	        pass
	with open('X.pkl','rb') as f:
		X = pickle.load(f)
	with open('y.pkl','rb') as f:
		y = pickle.load(f)
#Generating the training and testing dataset
	
	count_vectorizer = CountVectorizer()
	X = count_vectorizer.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
	#Naive Bayes Classifier
	clf = LogisticRegression()
	clf.fit(X_train,y_train)

	def pre(x):
		data1 = str(x)
		data1 = remove_new_lines(data1)
		data1 = remove_stop_words(data1)
		data1 = strip(data1)
		data1 = remove_weird(data1)
		data1 = np.array(data1).reshape(-1)
		vect = count_vectorizer.transform(data1)
		my_prediction1 = clf.predict(vect)
		return my_prediction1
	pred0 = pre(refined_tweets[0])
	pred1 = pre(refined_tweets[1])
	pred2 = pre(refined_tweets[2])
	pred3 = pre(refined_tweets[3])
	pred4 = pre(refined_tweets[4])	
	return render_template('twitter.html', des0 = refined_tweets[0],des1=refined_tweets[1],des2= refined_tweets[2], des3 = refined_tweets[3], des4 = refined_tweets[4], pred0=pred0,pred1=pred1,pred2=pred2,pred3=pred3,pred4=pred4)

@app.route('/news')
def news():
	with open('X.pkl','rb') as f:
		X = pickle.load(f)
	with open('y.pkl','rb') as f:
		y = pickle.load(f)
#Generating the training and testing dataset
	
	count_vectorizer = CountVectorizer()
	X = count_vectorizer.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
	#Naive Bayes Classifier
	clf = LogisticRegression()
	clf.fit(X_train,y_train)

	apikey = 'c9c0b7a1fc944a02bdadda8c09dace91'
	a = Articles(API_KEY=apikey)
	data = a.get(source="abc-news-au", sort_by='top')
	data = pd.DataFrame.from_dict(data)
	data = pd.concat([data.drop(['articles'], axis=1), data['articles'].apply(pd.Series)], axis=1)
	description = data['description']
	def pre(x):
		data1 = str(x)
		data1 = remove_new_lines(data1)
		data1 = remove_stop_words(data1)
		data1 = strip(data1)
		data1 = remove_weird(data1)
		data1 = np.array(data1).reshape(-1)
		vect = count_vectorizer.transform(data1)
		my_prediction1 = clf.predict(vect)
		return my_prediction1
	pred0 = pre(description[0])
	pred1 = pre(description[1])
	pred2 = pre(description[2])
	pred3 = pre(description[3])
	pred4 = pre(description[4])	

	return render_template('news.html', des0 = description[0],des1=description[1],des2= description[2], des3 = description[3], des4 = description[4], pred0=pred0,pred1=pred1,pred2=pred2,pred3=pred3,pred4=pred4)

@app.route('/predict',methods=['POST'])
def predict():	
	with open('X.pkl','rb') as f:
		X = pickle.load(f)
	with open('y.pkl','rb') as f:
		y = pickle.load(f)
#Generating the training and testing dataset
	
	count_vectorizer = CountVectorizer()
	X = count_vectorizer.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
	#Naive Bayes Classifier
	clf = LogisticRegression()
	clf.fit(X_train,y_train)

	if request.method == 'POST':
		comment = request.form['comment']
		data = str(comment)
		data = remove_new_lines(data)
		data = remove_stop_words(data)
		data = strip(data)
		data = remove_weird(data)
		data = np.array(data).reshape(-1)
		vect = count_vectorizer.transform(data)
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
