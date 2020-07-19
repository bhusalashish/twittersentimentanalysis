# twittersentimentanalysis
<b>**Twitter Sentimental Analysis Using Machine Learning**</b>
<br>
<p>Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.
It is the most common text classification tool that analyses an incoming message and tells whether the underlying sentiment is positive, negative our neutral.</p>
<br>
We start off our project by importing all the required packages.
After that we read our dataset with the help of pandas library.
Since it is a supervised learning task we are provided with a training data set which consists of Tweets labeled with “1” or “0” and a test data set without labels.
<br>
The given data sets are comprised of very much unstructured tweets which should be preprocessed to make an NLP model. So, after tokenizing we will process our data to get desired dataset for training our model.
<br>
<li>
	<ul>Removal of punctuations.</ul>
	<ul>Removal of commonly used words</ul>
	<ul>Normalization of words.</ul>
</li>
<br>
Before we let our data to train we have to numerically represent the preprocessed data. For this we use CountVectorization.
<br>
So, after vectorizing our sting data to numerical values we feedit to our machine learning model.
We choose naive bayes classifier for this binary classification since it is the most common algorithm used in NLP.
<br>
The next step is to split our dataset in to training data and testing data. We will do so by using train_test_split from sklearn. Then we fit our dataset into our model. Once our model is trained in out training dataset we use it to predict the values for the test dataset.
<br>
The final step is to analyse the result with the help of accuracy, recall, precision and F-Measure. 
