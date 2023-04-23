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

## Conclusion
The Fine-Grained Fake News Detection System is a powerful tool for identifying fake news articles with high accuracy. Its combination of machine learning algorithms and NLP techniques allows it to analyze the content of news articles and identify features that are indicative of fake news. By using this system, you can help prevent the spread of misinformation and ensure that news articles are accurate and reliable.

## References
- Wang, W., Yang, N., Wei, F., Chang, B., & Zhou, M. (2017). [Liar, liar pants on fire: A new benchmark dataset for fake news detection](https://www.aclweb.org/anthology/P17-2067/). In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 422-426).
- Augenstein, I., Rocktäschel, T., Vlachos, A., & Bontcheva, K. (2019). [A retrospective analysis of the fake news challenge stance detection task](https://www.aclweb.org/anthology/N19-1134/). In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 1316-1327).
