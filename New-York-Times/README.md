These are the files for the New York Times Agility exercise.

I have included his example of NLTK as well.

Professor Given Documents:
1) data_train.txt
2) data_valid.txt
3) labels_train_original.txt
4) labels_valid_original.txt
5) Task Description.txt
6) NLP - Econ 9-12-2018.py

Student Provided Documents:
1) data_train_art.txt  -- Periods at the end of each article
2) labels_train_numbers.txt  --  The labels converted to digits
3) hw.py  -- Spyder code for quick original analysis (No NLTK)

###########################################################################
Task Description 

1 Overview 

In this task, you will apply the machine learning techniques to a real-world dataset. Academia and training courses focus on applying a specific algorithm to carefully preprocessed toy datasets. This task will require that you make decisions such as which classifier to use, which features to use, how to train the classifier, etc. You will find that real world datasets are more difficult to work with, and that careful tuning is required to get optimal performance. 

2 Dataset 
In this project you will work with a labeled dataset of news articles. 

- Each instance in the dataset consists of the text of a New York Times article published between 2000 and 2003. Each article comes from one of four newspaper sections: News, Classifieds, Opinion, and Features. This dataset has been culled from the Linguistic Data Consortium's New York Times Annotated Corpus. <p>
- For your convenience the text has been slightly preprocessed: punctuation and formatting (such as paragraphs) have been removed, and all characters have been converted to lower case. <p>
- The articles are stored in data train.txt and data valid.txt. Each line in the file is a single article. Each article consists of a space-separated list of words. <p>
- The labels (the sections the articles came from) are stored in the labels train.txt and labels valid.txt. <p>
- Each line in the file is a single label. The nth line in the labels train.txt file contains the label for the article stored on the nth line of the data train.txt file. <p>
- The labels are numerical to make it easier to use them, but we have also provided the original text labels in labels train original.txt and labels test original.txt. The mapping is: <p>

1. News: 0 
2. Opinion: 1 
3. Classifieds: 2 
4. Features: 3 

YOUR TASK is to write a classifier which can correctly predict the label of a given article. You will need to implement some more sophisticated methods of feature generation, classification, or both. For instance, you might experiment with various techniques to combat over fitting, or you might use a dimensionality reduction technique to generate better features. 

3 Deliverables <p>
Create a document that describes how you solved the problem. It should contain detailed discussion about which methods you tried, results on their comparative performance, and a discussion about which methods worked best and why. It should follow the following simple outline: <p>
Introduction, <p>
Suggested Techniques 
A few classification techniques you may want to review/investigate: <p>
	1. Logistic regression <p>
	2. Neural networks <p>
	3. k-nearest neighbors <p>
	4. Support vector machines <p>
A few techniques for generating features that you may want to investigate: <p>
	1. Dimensionality reduction <p>
	2. The “bag of words" model <p>
	3. n-gram models (including smoothing techniques) <p>
	4. Other kinds of vector space models (see http://www.jair.org/media/2934/live-2934-4846-jair.pdf for several types of such models) <p>
	5. TF-IDF <p>
