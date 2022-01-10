# Word2Vec for Amazon Reviews
Three models (RNN, GRU and MLP) with an embedding layer for Amazon Reviews (dataset can be downloaded [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz)) for sentiment analysis classification using PyTorch, NLTK and Scikit. Two distinct embeddings were compared; a pretrained Word2Vec model <code>"word2vec-google-news-300"</code> from Google, and another trained from scratch by me. As a baseline, each embedding is deployed on a myriad of models using Scikit such as Perceptron, SVM, Logistic Regression, and Naive Bayes. Then, both embeddings are used in an MLP, RNN, and a GRU model to compare which model and embedding yielded the best result.

## Data Pre-processing
As is known, models have difficulty learning meaningful patterns from raw, unprocessed data. Thus, I cleaned the data and ensured the data was balanced so that the learning has no problems. The following steps were taken to clean the data:
<ul>
  <li>Lowercase all the reviews</li>
  <li>Remove all HTML tags and URLs</li>
  <li>Remove non-English alphanumeric characters and extra spaces</li>
  <li>Expand contractions (won't became will not)</li>
  <li>Remove stop words</li>
  <li>Perform lemmatization</li>
</ul>

The average character length of the reviews were reduced from 309 to 183 characters.

Next, I randomly sampled 100k instances of each class (positive and negative sentiment) to be the final processed dataset. By doing this, the model will be fitted on a high-quality dataset.

## Building a Word2Vec model
Thankfully, building a w2v model is not difficult. I used gensim's Word2Vec implementation to learn 300-dim vector embeddings from the Amazon reviews. As expected, <code>king-man+woman</code> sure enough gives <code>queen</code>.

## Models
To establish a baseline accuracy for each embedding, Scikit's Perceptron and Linear SVC implementations were used without much tweaking. Next, a MLP, RNN, and GRU models were built in PyTorch to showcase the richness and versatility of the embeddings. However, due to the nature of input of MLPs, the input size is fixed and so the reviews embeddings were either averaged or truncated to the first 10.

## Results for My Word2Vec Embedding
Overall, the results are similar; however, the Google embedding is slightly better. I believe this is due to their richer dataset that was trained on and possibly better hyperparameters.

| Model | Accuracy |
| ----- | -------- |
| Perceptron | 79% |
| Linear SVC | 82% |
| MLP - Averaged Input Features | 84% |
| MLP - First 10 Input Features | 81% |
| RNN | 86% |
| GRU | 90% |

## Results for Google News Word2Vec Embedding
| Model | Accuracy |
| ----- | -------- |
| Perceptron | 80% |
| Linear SVC | 82% |
| MLP - Averaged Input Features | 85% |
| MLP - First 10 Input Features | 83% |
| RNN | 88% |
| GRU | 93% |
