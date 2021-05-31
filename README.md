# Named Entity Recognition (NER) System Using RNN

### Goal
In this project I developed a NER system using Bi-LSTM Recurrent Neural Network architecture. 
 
### Description
* The dataset has total of 47959 sentences with total 1048575 words. There are 17 unique NER tags.

* The dataset is taken from Kaggle.  https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

* Most of the NER tags are "O" (Other) (84%) which means we are dealing with imbalanced dataset.

* Workflow:

    * Dataset is pre-processed
    * Created a vocabulary
    * Created mapping from words to integers
    * Extracted features
    * Bi-LSTM model is built
    * Model is run, best parameters are saved, evaluated the results
    * Obtained overall accuracy score of 99.3% and F1 Score of 80.4%.
    * Created a new sentence with related NER tags and made predictions on the new instance

### Project Files
* ner.ipynb - Project Notebook
* Dataset is not included because of size

### Libraries Used
*	numpy, pandas : for data manipulation
*	sklearn : for train/test split
*	tensorflow / keras : to build deep learning model
*	seqeval.metrics : to display metrics
*	pickle : to save objects

