# CSE508_Winter2023_Project_36
Fine-Grained Fake News Detection System developed by Group-36 of IIITD CSE508 Course (Information Retrieval. 

Contributors of the project are: 
- Anas Ahmad
- Anupam Narayan
- Ayush Sharma
- Harsh Goyal
- Shivam Jindal

## Introduction
This is a Fine-Grained Fake News Detection System that can classify news articles as real or fake. The system uses natural language processing (NLP) techniques to analyze the content of the articles and identify features that are indicative of fake news. The repository is originally developed for the Ccourse project of CSE508 Information Retrieval Course of IIIT Delhi.

## How it works
The Fine-Grained Fake News Detection System is built using a combination of machine learning algorithms and NLP techniques. The system is trained on a dataset containing short statements made by politicians, along with a label indicating whether the statement is true, mostly true, half true, mostly false, false, or pants on fire (which means a blatant lie). The statements were fact-checked by PolitiFact, a fact-checking organization in the United States. The link to data set is here
- Wang, W., Yang, N., Wei, F., Chang, B., & Zhou, M. (2017). "[LIAR: A LIar dataset for fact-checking research](https://www.aclweb.org/anthology/P17-2067.pdf)". In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 425-430). Vancouver, Canada: Association for Computational Linguistics.


The system then takes in a new statement as input and uses its learned knowledge to classify it as real or fake. The system analyzes the content of the article, looking for features such as sensational language, biased reporting, and inconsistent facts. Based on these features, the system assigns a probability score to the article, indicating the likelihood that it is fake news.



## Getting Started
To get started with the Fine-Grained Fake News Detection System, you will need to have Python 3 installed on your machine. You will also need to install the following Python packages:

- pandas
- numpy
- scikit-learn
- NLTK
- TensorFlow
- Keras
- transformers

Once you have these packages installed, you can download the Fine-Grained Fake News Detection System code from this GitHub repository. You will also need to download the dataset of labeled real and fake news articles, which is available for free online.

## Using the System
To use the Fine-Grained Fake News Detection System, you can run the any of the notebook files in your Python environment or Jupyter Notebook environment. Pro Tip - Use Google Colab for best hassle free experience

You can also modify the system's parameters and hyperparameters to improve its performance. For example, you can adjust the training data, change the NLP techniques used, or experiment with different machine learning algorithms.

# Acess to submissions
## [Baseline](https://github.com/harsh20562/CSE508_Winter2023_Project_36/tree/main/Baseline)
We established our problem statement and found the literture to take motivation and inspiration from and tried to mimick the baseline results implemented in them. We got familar with the dataset and learnt about the domain knowledge of the project we are working on. We used TFIDF (compute TFIDF matrix for the given text data) and CountVectorizer (makes the raw count matrix for the given text data) methods to convert textual data to numerical data and fed the numerical data to various machine learning models like Logistic Regression, SVMs, Decision Trees, Random Forests, etc.


## [Mid Project Review](https://github.com/harsh20562/CSE508_Winter2023_Project_36/tree/main/Mid%20Project%20Review)
Here we updated our problem statement and along with that our approach to solve it and provided the updated baseline results we developed along with the evidence of improvements over last deadline. We calculated POS taggings for the given text data and used TFIDF and CountVectorizer on the POS taggings to convert textual data to numerical data and fed this data to various machine learning models like Logistic Regression, Multi Layer Perceptron, etc.
We computed non-contextual word embeddings using GloVe embeddings which converts textual data to 100 dimensional vector. We also computed contextual embeddings using sentence transformer model all-MiniLM-L6-v2 which converts textual data to 384 dimensional vector for each token based on the context of the sentence. We gave non-contextual and contextual word embeddings as input to various machine learning models.


## [Final Submission](https://github.com/harsh20562/CSE508_Winter2023_Project_36/tree/main/Final%20Submission)
Here we proposed and executed the fully completed final methods for our problems. More specifically it contains models LSTM, CNN and BiLSTMs with following two approaches of inputs statements and statements with POS
- RNNs - We first tokenize the statements using the BERT tokenizer, and trains an RNN-based BERT model to predict the labels. The model consists of a BERT layer followed by an RNN layer The RNN part of the code is responsible for processing the output of the BERT layer and producing a final output for the model. Specifically, the RNN layer used in this code is a Bidirectional LSTM (Long Short-Term Memory) layer. The model is trained for 3 epochs using a batch size of 32, and the test accuracy is reported at the end. 
- CNN / LSTM / Bi-LSTM - For all these models, we created POS encodings for the preprocessed data. We passed the textual data and POS encodings to Embedding() layer to get the embeddings and passed these embeddings as input to the these models. We used categorical_crossentropy as loss function. We ran these models using two approaches - only giving embeddings of statement as input and giving embeddings of text as well as POS encodings as input to these models.


## Conclusion
The Fine-Grained Fake News Detection System is a powerful tool for identifying fake news articles with high accuracy. Its combination of machine learning algorithms and NLP techniques allows it to analyze the content of news articles and identify features that are indicative of fake news. By using this system, you can help prevent the spread of misinformation and ensure that news articles are accurate and reliable.

## References
- Wang, W., Yang, N., Wei, F., Chang, B., & Zhou, M. (2017). [Liar, liar pants on fire: A new benchmark dataset for fake news detection](https://www.aclweb.org/anthology/P17-2067/). In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 422-426).
- Augenstein, I., Rocktäschel, T., Vlachos, A., & Bontcheva, K. (2019). [A retrospective analysis of the fake news challenge stance detection task](https://www.aclweb.org/anthology/N19-1134/). In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 1316-1327).
