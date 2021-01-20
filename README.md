# Stackroute

The aim of the code is to read a library of documents, preprocess them, find similarity amongst them and recommend a similar document. The file app.py contains the python3 code to do this.

REQUIREMENTS TO RUN THE PYTHON CODE:-
numpy
nltk
gensim
matplotlib

The code executes the following pipeline:-
1) Scan the dir for all text documents->
The code scans the directory location for all available text documents and stores them in a dict with file names as the key.
2) Preprocess each document->
A standard text data clean-up procedure is followed with the followed step-
	a) Remove stop words
	Frequent stop words are removed from each documents. The stopwords list is imported from nltk and a few punctuations are added to the list.
	b) Generate POS tags
	Part-of-speech tags are generated for each of the remaining words. This is done to help nltk's WordNetLemmatizer to do it's task.
	c) Lemmatize the words
	The words are reduced to their base form using nltk's implementation of WordNetLemmatizer. A better or custom lemmatizer might help us get better results.
Generate document embeddings->
Dense vector embeddings are generated using gensim's Doc2Vec library.
	a) Define vectorizing model
	The vectorizer is defined to run in the distributed memory mode to preserve the contextual ordering of words with a learning rate of 0.025. Some hyper-parameter tuning might help us in getting a better and/or faster performance. So would a customised implementation of the NN for this specific task.
	b) Train vectorizing model
	The model is trained on the preprocessed corpus for 50 epoches. We can further check for the fitness of the model and tune this accordingly. 
3) Generate and save similarity matrix->
The trained model can give use a measure of similarity between two documents. The code iterates pair-wise over all combinations of the documents and generates a corresponding heat-map of similarty using matplotlib and saves it in the "figures" folder. For sanity's sake, the similarity score of a document with itself is explicitly set to 0 instead of 1.
4) Recommend most similar document->
For each document, the document with the highest similarity is found from the heat-map and recommended.
