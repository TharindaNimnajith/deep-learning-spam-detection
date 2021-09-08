#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Detection with Deep Learning
# 
# Short Message Service (SMS) is being heavily used as a way of communication. However, SMS spams have targeted most mobile phone users recently. In some cases, SMS spams contain malicious activities such as smishing. Smishing (SMS + Phishing) is a cyber-security threat for mobile users aimed at deceiving them via SMS spam messages that may include a link or malicious software or both. The attackers attempt to steal users' secret and sensitive information, like credit card numbers, bank account details, and passwords.
# 
# The filtration of SMS spams in smartphones is still not very robust compared to the filtration of email spams. The state-of-the-art methodologies based on Deep Learning can be utilized for solving this binary classification problem of SMS spam detection. We have used the two deep neural network architectures namely Long Short-Term Memory (LSTM) and DenseNet (Densely Connected Convolutional Neural Network (CNN)) for this purpose.

# In[1]:


# magic function that renders the figure in a notebook instead of
# displaying a dump of the figure object
# sets the backend of matplotlib to the 'inline' backend
# with this backend, the output of plotting commands is displayed
# inline within frontends like the Jupyter notebook, directly below
# the code cell that produced it
# the resulting plots will then also be stored in the notebook document
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# creating a new directory named plots
get_ipython().system('mkdir plots')

# creating a new directory named models
get_ipython().system('mkdir models')

# creating a new directory named processed_datasets
get_ipython().system('mkdir processed_datasets')


# In[3]:


# importing warnings library to handle exceptions, errors, and warning
# of the program
import warnings

# ignoring potential warnings of the program
warnings.filterwarnings('ignore')


# In[4]:


# importing pandas library to perform data manipulation and analysis
import pandas as pd

# configuring the pandas dataframes to show all columns
pd.options.display.max_columns = None

# configuring the pandas dataframes to increase the maximum column width
pd.options.display.max_colwidth = 150


# In[5]:


# downloading UCI SMS Spam Collection dataset
get_ipython().system('wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip')


# In[6]:


# extracting the downloaded dataset
get_ipython().system('unzip /content/smsspamcollection.zip')


# In[7]:


# listing files and directories
get_ipython().system('ls')


# # SMS Spam Collection Dataset
# 
# The SMS Spam Collection Dataset was downloaded from UCI datasets. It contains 5,574 SMS phone messages. The data were collected for the purpose of mobile phone SMS text message spam research and have already been labeled as either spam or ham.
# 
# Link to the dataset - http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

# In[8]:


# importing SMSSpamCollection dataset to a pandas dataframe
sms_spam_dataframe = pd.read_csv('/content/SMSSpamCollection',
                                 sep='\t',
                                 header=None,
                                 names=['class', 'sms_message'])
sms_spam_dataframe


# # Dataset Analysis and Data Preprocessing

# In[9]:


# printing the columns of the dataframe
sms_spam_dataframe.columns


# In[10]:


# changing the order of the dataframe columns for better visualization
sms_spam_dataframe = sms_spam_dataframe[['sms_message', 'class']]
sms_spam_dataframe


# In[11]:


# displaying the dimensionality of the dataframe
sms_spam_dataframe.shape


# In[12]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values,
# and memory usage
sms_spam_dataframe.info()


# In[13]:


# generating descriptive statistics of the dataframe
sms_spam_dataframe.describe()


# In[14]:


# generating descriptive statistics for each class of the dataframe
# T property is used to transpose index and columns of the dataframe
sms_spam_dataframe.groupby('class').describe().T


# In[15]:


# checking for missing or null values in the dataframe
dataframe_null = sms_spam_dataframe[sms_spam_dataframe.isnull().any(axis=1)]
dataframe_null


# In[16]:


# printing the number of rows with any missing or null values
# in the dataframe
dataframe_null.shape[0]


# In[17]:


# removing the missing or null values from the dataframe if exist
sms_spam_dataframe = sms_spam_dataframe[sms_spam_dataframe.notna().all(axis=1)]

# printing the count of null values in class and sms_message
# columns of the dataframe
sms_spam_dataframe[['class', 'sms_message']].isnull().sum()


# In[18]:


# importing pyplot from matplotlib library to create interactive
# visualizations
import matplotlib.pyplot as plt

# importing seaborn library which is built on top of matplotlib to
# create statistical graphics
import seaborn as sns

# plotting the heatmap for missing or null values in the dataframe
sns.heatmap(sms_spam_dataframe.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
plt.title('Null Values Detection Heat Map')
plt.savefig('plots/null_detection_heat_map.png',
            facecolor='white')
plt.show()


# In[19]:


# importing missingno library
# used to understand the distribution of missing values through
# informative visualizations
# visualizations can be in the form of heat maps or bar charts
# used to observe where the missing values have occurred
# used to check the correlation of the columns containing the missing
# with the target column
import missingno as msno

# plotting a matrix visualization of the nullity of the dataframe
fig = msno.matrix(sms_spam_dataframe)
fig_copy = fig.get_figure()
fig_copy.savefig('plots/msno_matrix.png',
                 bbox_inches='tight')
fig


# In[20]:


# plotting a seaborn heatmap visualization of nullity correlation
# in the dataframe
fig = msno.heatmap(sms_spam_dataframe)
fig_copy = fig.get_figure()
fig_copy.savefig('plots/msno_heatmap.png',
                 bbox_inches='tight')
fig


# In[21]:


# detecting duplicate rows exist in the dataframe before cleaning
duplicated_records = sms_spam_dataframe[sms_spam_dataframe.duplicated()]
duplicated_records


# In[22]:


# checking the number of duplicate rows exist in the dataframe
# before cleaning
sms_spam_dataframe.duplicated().sum()


# In[23]:


# removing the duplicate rows from the dataframe if exist
sms_spam_dataframe = sms_spam_dataframe.drop_duplicates()
sms_spam_dataframe


# In[24]:


# checking the number of duplicate rows exist in the dataframe
# after cleaning
sms_spam_dataframe.duplicated().sum()


# In[25]:


# displaying the dimensionality of the dataframe
sms_spam_dataframe.shape


# In[26]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values,
# and memory usage
sms_spam_dataframe.info()


# In[27]:


# generating descriptive statistics of the dataframe
sms_spam_dataframe.describe()


# In[28]:


# generating descriptive statistics for each class of the dataframe
# T property is used to transpose index and columns of the dataframe
sms_spam_dataframe.groupby('class').describe().T


# In[29]:


# saving cleaned dataset to a csv file
file_name = 'processed_datasets/cleaned_dataset.csv'
sms_spam_dataframe.to_csv(file_name,
                          encoding='utf-8',
                          index=False)

# loading dataset from the saved csv file to a pandas dataframe
cleaned_sms_spam_dataframe = pd.read_csv(file_name)
cleaned_sms_spam_dataframe


# In[30]:


# importing set of stopwords from wordcloud library
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

# printing number of stopwords defined in wordcloud library
len(stopwords)


# In[31]:


# importing random library
# used for generating random numbers
import random

# printing 10 random values of stopwords set
for i, val in enumerate(random.sample(stopwords, 10)):
    print(val)


# In[32]:


# importing WordCloud object for generating and drawing
# wordclouds from wordcloud library
from wordcloud import WordCloud


# defining a function to return the wordcloud for a given text
def plot_wordcloud(text):
    wordcloud = WordCloud(width=600,
                          height=300,
                          background_color='black',
                          stopwords=stopwords,
                          max_font_size=50,
                          colormap='Oranges').generate(text)
    return wordcloud


# In[33]:


# extracting the data instances with class label 'ham'
ham_dataframe = cleaned_sms_spam_dataframe[cleaned_sms_spam_dataframe['class'] == 'ham']
ham_dataframe


# In[34]:


# creating numpy list to visualize using wordcloud
ham_sms_message_text = ' '.join(ham_dataframe['sms_message'].to_numpy().tolist())

# generating wordcloud for ham sms messages
ham_sms_wordcloud = plot_wordcloud(ham_sms_message_text)
plt.figure(figsize=(16, 10))
plt.imshow(ham_sms_wordcloud,
           interpolation='bilinear')
plt.axis('off')
plt.title('Ham SMS Wordcloud')
plt.savefig('plots/ham_wordcloud.png',
            facecolor='white')
plt.show()


# In[35]:


# extracting the data instances with class label 'spam'
spam_dataframe = cleaned_sms_spam_dataframe[cleaned_sms_spam_dataframe['class'] == 'spam']
spam_dataframe


# In[36]:


# creating numpy list to visualize using wordcloud
spam_sms_message_text = ' '.join(spam_dataframe['sms_message'].to_numpy().tolist())

# generating wordcloud for spam sms messages
spam_sms_wordcloud = plot_wordcloud(spam_sms_message_text)
plt.figure(figsize=(16, 10))
plt.imshow(spam_sms_wordcloud,
           interpolation='bilinear')
plt.axis('off')
plt.title('Spam SMS Wordcloud')
plt.savefig('plots/spam_wordcloud.png',
            facecolor='white')
plt.show()


# In[37]:


# printing count of values in each class of the dataframe
cleaned_sms_spam_dataframe['class'].value_counts()


# In[38]:


# plotting the distribution of target values
fig = plt.figure()
lbl = ['Ham (0)', 'Spam (1)']
pct = '%1.0f%%'
ax = cleaned_sms_spam_dataframe['class'].value_counts().plot(kind='pie',
                                                             labels=lbl,
                                                             autopct=pct)
ax.yaxis.set_visible(False)
plt.title('Distribution of Ham and Spam SMS')
plt.legend()
fig.savefig('plots/ham_spam_pie_chart.png',
            facecolor='white')
plt.show()


# In[39]:


# downsampling is a process where you randomly delete some of the
# observations from the majority class so that the numbers in majority
# and minority classes are matched
# after downsampling the ham messages (majority class), there are now
# 747 messages in each class
downsampled_ham_dataframe = ham_dataframe.sample(n=len(spam_dataframe),
                                                 random_state=44)
downsampled_ham_dataframe


# In[40]:


# printing the dimensions of spam and downsampled ham dataframes
print('Spam dataframe shape:', spam_dataframe.shape)
print('Ham dataframe shape:', downsampled_ham_dataframe.shape)


# In[41]:


# merging the two dataframes (spam + downsampled ham dataframes)
merged_dataframe = pd.concat([downsampled_ham_dataframe, spam_dataframe])
merged_dataframe = merged_dataframe.reset_index(drop=True)
merged_dataframe


# In[42]:


# printing count of values in each class of the merged dataframe
merged_dataframe['class'].value_counts()


# In[43]:


# plotting the distribution of target values after downsampling
fig = plt.figure()
lbl = ['Ham (0)', 'Spam (1)']
pct = '%1.0f%%'
ax = merged_dataframe['class'].value_counts().plot(kind='pie',
                                                   labels=lbl,
                                                   autopct=pct)
ax.yaxis.set_visible(False)
plt.title('Distribution of Ham and Spam SMS after Downsampling')
plt.legend()
fig.savefig('plots/ham_spam_pie_chart_after_downsampling.png',
            facecolor='white')
plt.show()


# In[44]:


# inserting a new column called 'label' to the merged dataframe
# if class is 'ham' label = 0
# if class is 'spam' label = 1
merged_dataframe['label'] = merged_dataframe['class'].map({'ham': 0, 'spam': 1})
merged_dataframe


# In[45]:


# inserting a new column called 'length' to the merged dataframe
# the column contains the number of characters of the sms_message text
merged_dataframe['length'] = merged_dataframe['sms_message'].apply(len)
merged_dataframe


# In[46]:


# displaying the first 5 rows of the dataframe
merged_dataframe.head()


# In[47]:


# displaying the last 5 rows of the dataframe
merged_dataframe.tail()


# In[48]:


# displaying the dimensionality of the dataframe
merged_dataframe.shape


# In[49]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values,
# and memory usage
merged_dataframe.info()


# In[50]:


# generating descriptive statistics of the dataframe
merged_dataframe.describe().round(2)


# In[51]:


# generating descriptive statistics for each class of the dataframe
# T property is used to transpose index and columns of the dataframe
merged_dataframe.groupby('label').describe().T


# In[52]:


# generating descriptive sms text length statistics by label types
merged_dataframe.groupby('label')['length'].describe().round(2)


# In[53]:


# plotting a univariate distribution of observations for sms lengths
sns.distplot(merged_dataframe['length'].values)
plt.title('SMS Lengths Distribution')
plt.xlabel('SMS Length')
plt.savefig('plots/sms_length.png',
            facecolor='white')
plt.show()


# In[54]:


# saving merged dataset to a csv file
file_name = 'processed_datasets/merged_dataset.csv'
merged_dataframe.to_csv(file_name,
                        encoding='utf-8',
                        index=False)

# loading dataset from the saved csv file to a pandas dataframe
merged_dataframe = pd.read_csv(file_name)
merged_dataframe


# In[55]:


# assigning attributes (features) to X
X = merged_dataframe['sms_message']
X


# In[56]:


# assigning label (target) to y
y = merged_dataframe['label'].values
y


# In[57]:


# importing train_test_split from scikit-learn library
from sklearn.model_selection import train_test_split

# splitting data into random train and test subsets
# train set - 80%, test set - 20%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=443)

# printing the dimension of train features dataframe
print('Shape of train features dataframe:', X_train.shape)

# printing the dimension of train target dataframe
print('Shape of train target dataframe:', y_train.shape)

# printing the dimension of test features dataframe
print('Shape of test features dataframe:', X_test.shape)

# printing the dimension of test target dataframe
print('Shape of test target dataframe:', y_test.shape)


# In[58]:


# displaying train features dataframe
X_train


# In[59]:


# displaying train target dataframe
y_train


# In[60]:


# displaying test features dataframe
X_test


# In[61]:


# displaying test target dataframe
y_test


# In[62]:


# defining pre-processing hyperparameters

# oov_token defines the out of vocabulary token
# oov_token will be added to word index in the corpus which is used to
# build the model
# this is used to replace out of vocabulary words (words that are not
# in our corpus) during text_to_sequence calls.
oov_token = '<OOV>'

# vocabulary_size indicates the maximum number of unique words to
# tokenize and load in training and testing data
vocabulary_size = 500


# In[63]:


# importing Tokenizer from keras library
# tensorflow is a free and open-source software library for machine
# learning used across a range of machine learning related tasks
# focus on training and inference of deep neural networks
# keras is a high-level api of tensorflow
# keras.preprocessing.text provides keras data preprocessing utils
# to pre-process datasets with textual data before they are fed to the
# machine learning model
from tensorflow.keras.preprocessing.text import Tokenizer

# Tokenizer allows to vectorize a text corpus, by turning each text into
# either a sequence of integers (each integer being the index of a token
# in a dictionary) or into a vector where the coefficient for each token
# could be binary, based on word count, based on tf-idf
tokenizer = Tokenizer(num_words=vocabulary_size,
                      char_level=False,
                      oov_token=oov_token)

# updating internal vocabulary based on a list of text required before
# using texts_to_sequences
tokenizer.fit_on_texts(X_train)


# In[64]:


# getting the word_index
word_index = tokenizer.word_index
word_index


# In[65]:


# printing length of the word index
len(word_index)


# In[66]:


# transforming each text in train data to a sequence of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)

# printing the first sequence
X_train_sequences[0]


# In[67]:


# getting lengths of each generated sequences of integers
# in train data
x_train_length_of_sequence = [len(sequence) for sequence in X_train_sequences]

# printing the length of the first sequence
x_train_length_of_sequence[0]


# In[68]:


# importing numpy library
# used to perform fast mathematical operations over python arrays
# and lists
import numpy as np

# printing maximum length of a sequence in the train data
np.max(x_train_length_of_sequence)


# In[69]:


# plotting a univariate distribution of observations for
# sequence lengths of train data
sns.distplot(x_train_length_of_sequence)
plt.title('Train Data Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/train_sequence_length.png',
            facecolor='white')
plt.show()


# In[70]:


# transforming each text in test data to a sequence of integers
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# printing the first sequence
X_test_sequences[0]


# In[71]:


# getting lengths of each generated sequences of integers
# in test data
x_test_length_of_sequence = [len(sequence) for sequence in X_test_sequences]

# printing the length of the first sequence
x_test_length_of_sequence[0]


# In[72]:


# printing maximum length of a sequence in the test data
np.max(x_test_length_of_sequence)


# In[73]:


# plotting a univariate distribution of observations for sequence
# lengths of test data
sns.distplot(x_test_length_of_sequence)
plt.title('Test Data Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/test_sequence_length.png',
            facecolor='white')
plt.show()


# In[74]:


# defining pre-processing hyperparameters

# maximum_length indicates the maximum number of words considered
# in a text
maximum_length = 50

# truncating_type indicates removal of values from sequences larger
# than maxlen, either at the beginning ('pre') or at the end ('post')
# of the sequences
truncating_type = 'post'

# padding_type indicates pad either before ('pre') or after ('post')
# each sequence
padding_type = 'post'


# In[75]:


# importing utilities for preprocessing sequence data from
# keras library
from tensorflow.keras.preprocessing.sequence import pad_sequences

# padding on train data
# pad_sequences pads sequences to the same length
# padding='post' to pad after each sequence
X_train_padded = pad_sequences(X_train_sequences,
                               maxlen=maximum_length,
                               padding=padding_type,
                               truncating=truncating_type)

# printing the first padded sequence
X_train_padded[0]


# In[76]:


# getting lengths of each padded sequences of integers in train
# data
x_train_length_of_padded_sequence = [len(sequence) for sequence in X_train_padded]

# printing the length of the first padded sequence
x_train_length_of_padded_sequence[0]


# In[77]:


# printing maximum length of a padded sequence in the train data
np.max(x_train_length_of_padded_sequence)


# In[78]:


# printing the dimension of padded training dataframe
X_train_padded.shape


# In[79]:


# plotting a univariate distribution of observations for sequence
# lengths of train data after padding
sns.distplot(x_train_length_of_padded_sequence)
plt.title('Train Data Padded Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/train_padded_sequence_length.png',
            facecolor='white')
plt.show()


# In[80]:


# padding on test data
# pad_sequences pads sequences to the same length
# padding='post' to pad after each sequence
X_test_padded = pad_sequences(X_test_sequences,
                              maxlen=maximum_length,
                              padding=padding_type,
                              truncating=truncating_type)

# printing the first padded sequence
X_test_padded[0]


# In[81]:


# getting lengths of each padded sequences of integers in test
# data
x_test_length_of_padded_sequence = [len(sequence) for sequence in X_test_padded]

# printing the length of the first padded sequence
x_test_length_of_padded_sequence[0]


# In[82]:


# printing maximum length of a padded sequence in the test data
np.max(x_test_length_of_padded_sequence)


# In[83]:


# printing the dimension of padded test dataframe
X_test_padded.shape


# In[84]:


# plotting a univariate distribution of observations for
# sequence lengths of test data after padding
sns.distplot(x_test_length_of_padded_sequence)
plt.title('Test Data Padded Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/test_padded_sequence_length.png',
            facecolor='white')
plt.show()


# # LSTM Model
# 
# Long Short Term Memory (LSTM) is a special kind of Recurrent Neural Network (RNN). LSTM models are explicitly designed to avoid the long-term dependency problem by remembering information for long periods of time.
# 
# ![index.png](data:image/png;base64,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)

# In[85]:


# LSTM network architecture hyperparameters

# SpatialDropout1D is used to dropout the embedding layer which helps
# to drop entire 1D feature maps instead of individual elements
# dropout_rate indicates the fraction of the units to drop for the
# linear transformation of the inputs
dropout_rate = 0.2

# n_lstm indicates the number of nodes in the hidden layers within the
# LSTM cell
no_of_nodes = 20

# embedding dimension indicates the dimension of the state space used for
# reconstruction
embedding_dimension = 16

# no_of_epochs indicates the number of complete passes through the
# training dataset
no_of_epochs = 30

# vocabulary_size indicates the maximum number of unique words to
# tokenize and load in training and testing data
vocabulary_size = 500


# In[86]:


# importing Sequential class from keras
# Sequential groups a linear stack of layers into a keras Model
from tensorflow.keras.models import Sequential

# importing Embedding class from keras layers api package
# turning positive integers (indexes) into dense vectors of fixed size
# this layer can only be used as the first layer in a model
from tensorflow.keras.layers import Embedding

# importing LSTM class from keras layers api package
# LSTM - Long Short-Term Memory layer
from tensorflow.keras.layers import LSTM

# importing Dense class from keras layers api package
# Dense class is a regular densely-connected neural network layer
# Dense implements the operation:
# output = activation(dot(input, kernel) + bias)
# activation is the element-wise activation function passed as
# the activation argument
# kernel is a weights matrix created by the layer
# bias is a bias vector created by the layer
# (only applicable if use_bias is True)
from tensorflow.keras.layers import Dense


# In[87]:


# LSTM model architecture

lstm_model = Sequential()

lstm_model.add(Embedding(vocabulary_size,
                         embedding_dimension,
                         input_length=maximum_length))

# return_sequences=True ensures that the LSTM cell returns all of the
# outputs from the unrolled LSTM cell through time
# if this argument is not used, the LSTM cell will simply provide the
# output of the LSTM cell from the previous step
lstm_model.add(LSTM(no_of_nodes,
                    dropout=dropout_rate,
                    return_sequences=True))

lstm_model.add(LSTM(no_of_nodes,
                    dropout=dropout_rate,
                    return_sequences=True))

# sigmoid is a non-linear and easy to work with activation function
# that takes a value as input and outputs another value between 0 and 1
lstm_model.add(Dense(1,
                     activation='sigmoid'))


# In[88]:


# compiling the model
# configuring the model for training
lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])


# In[89]:


# printing a string summary of the network
lstm_model.summary()


# In[90]:


# importing EarlyStopping class from callbacks module
# in keras library
# callbacks module includes utilities called at certain
# points during model training
# used to stop training when a monitored metric has
# stopped improving
from tensorflow.keras.callbacks import EarlyStopping

# monitoring the validation loss and if the validation loss is not
# improved after three epochs, then the model training is stopped
# it helps to avoid overfitting problem and indicates when to stop
# training before the deep learning model begins overfitting
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3)


# In[91]:


# training the LSTM model
history = lstm_model.fit(X_train_padded,
                         y_train,
                         epochs=no_of_epochs,
                         validation_data=(X_test_padded, y_test),
                         callbacks=[early_stopping],
                         verbose=2)


# In[92]:


# visualizing the history results by reading as a dataframe
metrics_lstm = pd.DataFrame(history.history)
metrics_lstm


# In[93]:


# renaming the column names of the dataframe
metrics_lstm.rename(columns={'loss': 'Training_Loss',
                             'accuracy': 'Training_Accuracy',
                             'val_loss': 'Validation_Loss',
                             'val_accuracy': 'Validation_Accuracy'},
                    inplace=True)
metrics_lstm


# In[94]:


# plotting the training and validation loss by number of epochs for
# the LSTM model
metrics_lstm[['Training_Loss', 'Validation_Loss']].plot()
plt.title('LSTM Model - Training and Validation Loss vs. Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['Training_Loss', 'Validation_Loss'])
plt.savefig('plots/lstm_loss_vs_epochs.png',
            facecolor='white')
plt.show()


# In[95]:


# plotting the training and validation accuracy by number of epochs for
# the LSTM model
metrics_lstm[['Training_Accuracy', 'Validation_Accuracy']].plot()
plt.title('LSTM Model - Training and Validation Accuracy vs. Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training_Accuracy', 'Validation_Accuracy'])
plt.savefig('plots/lstm_accuracy_vs_epochs.png',
            facecolor='white')
plt.show()


# In[96]:


# saving the trained LSTM model as an h5 file
# h5 is a file format to store structured data
# keras saves deep learning models in this format as it can easily store
# the weights and model configuration in a single file
lstm_path = 'models/lstm_model.h5'
lstm_model.save(lstm_path)
lstm_model


# In[97]:


# importing load_model function from keras to load a saved keras
# deep learning model
from tensorflow.keras.models import load_model

# loading the saved LSTM model
loaded_lstm_model = load_model(lstm_path)
loaded_lstm_model


# # LSTM Model Evaluation

# In[98]:


# evaluating the LSTM model performance on test data
# validation loss = 0.21674150228500366
# validation accuracy = 0.930610716342926
loaded_lstm_model.evaluate(X_test_padded,
                           y_test)


# In[99]:


# predicting labels of X_test data values on the basis of the
# trained model
y_pred_lstm = [1 if x[0][0] > 0.5 else 0 for x in loaded_lstm_model.predict(X_test_padded)]

# printing the length of the predictions list
len(y_pred_lstm)


# In[100]:


# printing the first 25 elements of the predictions list
y_pred_lstm[:25]


# In[101]:


# importing mean_squared_error from scikit-learn library
from sklearn.metrics import mean_squared_error

# mean squared error (MSE)
print('MSE :', mean_squared_error(y_test,
                                  y_pred_lstm))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted
# and actual value of variable
print('RMSE:', mean_squared_error(y_test,
                                  y_pred_lstm,
                                  squared=False))


# In[102]:


# importing mean_absolute_error from scikit-learn library
from sklearn.metrics import mean_absolute_error

# mean absolute error (MAE)
print('MAE:', mean_absolute_error(y_test,
                                  y_pred_lstm))


# In[103]:


# importing accuracy_score from scikit-learn library
from sklearn.metrics import accuracy_score

# accuracy
# ratio of the number of correct predictions to the total number of
# input samples
print('Accuracy:', accuracy_score(y_test,
                                  y_pred_lstm))


# In[104]:


# importing precision_recall_fscore_support from scikit-learn library
from sklearn.metrics import precision_recall_fscore_support

print('\t\t\tPrecision \t\tRecall \t\tF-Measure \tSupport')

# computing precision, recall, f-measure and support for each class
# with average='micro'
print('average=micro    -', precision_recall_fscore_support(y_test,
                                                            y_pred_lstm,
                                                            average='micro'))

# computing precision, recall, f-measure and support for each class
# with average='macro'
print('average=macro    -', precision_recall_fscore_support(y_test,
                                                            y_pred_lstm,
                                                            average='macro'))

# computing precision, recall, f-measure and support for each class
# with average='weighted'
print('average=weighted -', precision_recall_fscore_support(y_test,
                                                            y_pred_lstm,
                                                            average='weighted'))


# In[105]:


# importing classification_report from scikit-learn library
# used to measure the quality of predictions from a
# classification algorithm
from sklearn.metrics import classification_report

# report shows the main classification metrics precision, recall and
# f1-score on a per-class basis
print(classification_report(y_test,
                            y_pred_lstm))


# In[106]:


# importing confusion_matrix from scikit-learn library
from sklearn.metrics import confusion_matrix

# confusion matrix is a summarized table used to assess the performance
# of a classification model
# number of correct and incorrect predictions are summarized with their
# count according to each class
print(confusion_matrix(y_test,
                       y_pred_lstm))


# In[107]:


# importing plot_confusion_matrix from scikit-learn library

# plotting the confusion matrix
cm = confusion_matrix(y_test,
                      y_pred_lstm)
sns.heatmap(cm,
            annot=True,
            cbar=False,
            fmt='g')
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('plots/lstm_confusion_matrix.png',
            facecolor='white')
plt.show()


# In[108]:


# plotting scatter plot to visualize overlapping of predicted and
# test target data points
plt.scatter(range(len(y_pred_lstm)),
            y_pred_lstm,
            color='red')
plt.scatter(range(len(y_test)),
            y_test,
            color='green')
plt.title('LSTM - Predicted label vs. Actual label')
plt.xlabel('SMS Messages')
plt.ylabel('Label')
plt.savefig('plots/lstm_predicted_vs_real.png',
            facecolor='white')
plt.show()


# # Densely Connected CNN (DenseNet) Model
# 
# A DenseNet is a type of Convolutional Neural Network (CNN) that utilises dense connections between layers, where all layers with matching feature-map sizes are connected directly with each other. With the dense connections, higher accuracy is achieved with fewer parameters compared to a traditional CNN.
# 
# ![densenet_architecture.png](data:image/png;base64,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bj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KeUk4xwAAABxpRE9UAAAAAgAAAAAAAAECAAAAKAAAAQIAAAECAADAdZteALMAAEAASURBVHgB7J0FXFVpt8YfkFBCBTvB7u7u7u6xW8cYddSxu7u7x+7u7i4UFEEFJKRREIW71h7hQzj7UMoB73rvzznn7Hj33v9z7u97WO9az9ILpQEZQkAICAEhIASEgBAQAkJAxwT0RJjq+BuQywsBISAEhIAQEAJCQAgoBESYyg9BCAgBISAEhIAQEAJCIFEQEGGaKL4GuQkhIASEgBAQAkJACAgBEabyGxACQkAICAEhIASEgBBIFAREmCaKr0FuQggIASEgBISAEBACQkCEqfwGhIAQEAJCQAgIASEgBBIFARGmieJrkJsQAkJACAgBISAEhIAQEGEqvwEhIASEgBAQAkJACAiBREFAhGmi+BrkJoSAEBACQkAICAEhIAREmMpvQAgIASEgBISAEBACQiBREBBhmii+BrkJISAEhIAQEAJCQAgIARGm8hsQAkJACAgBISAEhIAQSBQERJgmiq9BbkIICAEhIASEgBAQAkJAhKn8BoSAEBACQkAICAEhIAQSBQERponia5CbEAJCQAgIASEgBISAEBBhKr8BISAEhIAQEAJCQAgIgURBQIRpovga5CaEgBAQAkJACAgBISAERJjKb0AICAEhIASEgBAQAkIgURAQYZoovga5CSEgBISAEBACQkAICAERpvIbEAJCQAgIASEgBISAEEgUBESYJoqvQW5CCAgBISAEhIAQEAJCQISp/AaEgBAQAkJACAgBISAEEgUBEaaJ4muQmxACQkAICAEhIASEgBAQYSq/ASEgBISAEBACQkAICIFEQUCEaaL4GuQmhIAQEAJCQAgIASEgBESYym9ACAgBISAEhIAQEAJCIFEQEGGaKL4GuQkhIASEgBAQAkJACAgBEabyGxACQkAICAEhIASEgBBIFAREmCaKr0FuQggIASEgBISAEBACQkCEqfwGhIAQEAJCQAgIASEgBBIFARGmieJrkJsQAkJACAgBISAEhIAQEGEqvwEhIASEgBAQAkJACAiBREFAhGmi+BrkJoSAEBACQkAICAEhIAREmNJvwMnJCVmyZJFfgxAQAkJACAgBISAEhIAOCYgwJfh58uSBnZ2dDr8GubQQ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)

# In[109]:


# Densely Connected CNN (DenseNet) architecture hyperparameters

# SpatialDropout1D is used to dropout the embedding layer which helps
# to drop entire 1D feature maps instead of individual elements
# dropout_rate indicates the fraction of the units to drop for the
# linear transformation of the inputs
dropout_rate = 0.2

# embedding dimension indicates the dimension of the state space used for
# reconstruction
embedding_dimension = 16

# no_of_epochs indicates the number of complete passes through the
# training dataset
no_of_epochs = 30

# vocabulary_size indicates the maximum number of unique words to
# tokenize and load in training and testing data
vocabulary_size = 500


# In[110]:


# importing Sequential class from keras
# Sequential groups a linear stack of layers into a keras Model
from tensorflow.keras.models import Sequential

# importing Embedding class from keras layers api package
# turning positive integers (indexes) into dense vectors of fixed size
# this layer can only be used as the first layer in a model
from tensorflow.keras.layers import Embedding

# importing GlobalAveragePooling1D class from keras layers api package
# used to perform global average pooling operation for temporal data
from tensorflow.keras.layers import GlobalAveragePooling1D

# importing Dropout class from keras layers api package
# used to apply Dropout to the input
# Dropout is one of the most effective and most commonly used
# regularization techniques for neural networks
# Dropout, applied to a layer, consists of randomly 'dropping out'
# (set to zero) a number of output features of the layer during training 
from tensorflow.keras.layers import Dropout

# importing Dense class from keras layers api package
# Dense class is a regular densely-connected neural network layer
# Dense implements the operation:
# output = activation(dot(input, kernel) + bias)
# activation is the element-wise activation function passed as
# the activation argument
# kernel is a weights matrix created by the layer
# bias is a bias vector created by the layer
# (only applicable if use_bias is True)
from tensorflow.keras.layers import Dense


# In[111]:


# Densely Connected CNN (DenseNet) model architecture

densenet_cnn_model = Sequential()

densenet_cnn_model.add(Embedding(vocabulary_size,
                                 embedding_dimension,
                                 input_length=maximum_length))

densenet_cnn_model.add(GlobalAveragePooling1D())

# Rectified Linear Activation Function (ReLU) is a piecewise linear
# function that will output the input directly if it is positive,
# otherwise, it will output zero
densenet_cnn_model.add(Dense(24,
                             activation='relu'))

densenet_cnn_model.add(Dropout(dropout_rate))

# sigmoid is a non-linear and easy to work with activation function
# that takes a value as input and outputs another value between 0 and 1
densenet_cnn_model.add(Dense(1,
                             activation='sigmoid'))


# In[112]:


# compiling the model
# configuring the model for training
densenet_cnn_model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


# In[113]:


# printing a string summary of the network
densenet_cnn_model.summary()


# In[114]:


# monitoring the validation loss and if the validation loss is not
# improved after three epochs, then the model training is stopped
# it helps to avoid overfitting problem and indicates when to stop
# training before the deep learning model begins overfitting
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3)


# In[115]:


# training the DenseNet CNN model
history = densenet_cnn_model.fit(X_train_padded,
                                 y_train,
                                 epochs=no_of_epochs,
                                 validation_data=(X_test_padded, y_test),
                                 callbacks=[early_stopping],
                                 verbose=2)


# In[116]:


# visualizing the history results by reading as a dataframe
densenet_cnn_metrics = pd.DataFrame(history.history)
densenet_cnn_metrics


# In[117]:


# renaming the column names of the dataframe
densenet_cnn_metrics.rename(columns={'loss': 'Training_Loss',
                                     'accuracy': 'Training_Accuracy',
                                     'val_loss': 'Validation_Loss',
                                     'val_accuracy': 'Validation_Accuracy'},
                            inplace=True)
densenet_cnn_metrics


# In[118]:


# plotting the training and validation loss by number of epochs for
# the DenseNet CNN model
densenet_cnn_metrics[['Training_Loss', 'Validation_Loss']].plot()
plt.title('DenseNet CNN Model - Training and Validation Loss vs. Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['Training_Loss', 'Validation_Loss'])
plt.savefig('plots/densenet_cnn_loss_vs_epochs.png',
            facecolor='white')
plt.show()


# In[119]:


# plotting the training and validation accuracy by number of epochs
# for the DenseNet CNN model
densenet_cnn_metrics[['Training_Accuracy', 'Validation_Accuracy']].plot()
plt.title('DenseNet CNN Model - Training and Validation Accuracy vs. Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training_Accuracy', 'Validation_Accuracy'])
plt.savefig('plots/densenet_cnn_accuracy_vs_epochs.png',
            facecolor='white')
plt.show()


# In[120]:


# saving the trained DenseNet CNN model as an h5 file
# h5 is a file format to store structured data
# keras saves deep learning models in this format as it can easily store
# the weights and model configuration in a single file
densenet_cnn_path = 'models/densenet_cnn_model.h5'
densenet_cnn_model.save(densenet_cnn_path)
densenet_cnn_model


# In[121]:


# loading the saved DenseNet CNN model
loaded_densenet_cnn_model = load_model(densenet_cnn_path)
loaded_densenet_cnn_model


# # Densely Connected CNN (DenseNet) Model Evaluation

# In[122]:


# evaluating the DenseNet CNN model performance on test data
# validation loss = 0.11119994521141052
# validation accuracy = 0.9732824563980103
loaded_densenet_cnn_model.evaluate(X_test_padded,
                                   y_test)


# In[123]:


# predicting labels of X_test data values on the basis of the
# trained model
y_pred_densenet_cnn = [1 if x[0] > 0.5 else 0 for x in loaded_densenet_cnn_model.predict(X_test_padded)]

# printing the length of the predictions list
len(y_pred_densenet_cnn)


# In[124]:


# printing the first 25 elements of the predictions list
y_pred_densenet_cnn[:25]


# In[125]:


# mean squared error (MSE)
print('MSE :', mean_squared_error(y_test,
                                  y_pred_densenet_cnn))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted
# and actual value of variable
print('RMSE:', mean_squared_error(y_test,
                                  y_pred_densenet_cnn,
                                  squared=False))


# In[126]:


# mean absolute error (MAE)
print('MAE:', mean_absolute_error(y_test,
                                  y_pred_densenet_cnn))


# In[127]:


# accuracy
# ratio of the number of correct predictions to the total number of
# input samples
print('Accuracy:', accuracy_score(y_test,
                                  y_pred_densenet_cnn))


# In[128]:


print('\t\t\tPrecision \t\tRecall \t\tF-Measure \tSupport')

# computing precision, recall, f-measure and support for each class
# with average='micro'
print('average=micro    -', precision_recall_fscore_support(y_test,
                                                            y_pred_densenet_cnn,
                                                            average='micro'))

# computing precision, recall, f-measure and support for each class
# with average='macro'
print('average=macro    -', precision_recall_fscore_support(y_test,
                                                            y_pred_densenet_cnn,
                                                            average='macro'))

# computing precision, recall, f-measure and support for each class
# with average='weighted'
print('average=weighted -', precision_recall_fscore_support(y_test,
                                                            y_pred_densenet_cnn,
                                                            average='weighted'))


# In[129]:


# report shows the main classification metrics precision, recall and
# f1-score on a per-class basis
print(classification_report(y_test,
                            y_pred_densenet_cnn))


# In[130]:


# confusion matrix is a summarized table used to assess the performance
# of a classification model
# number of correct and incorrect predictions are summarized with their
# count according to each class
print(confusion_matrix(y_test,
                       y_pred_densenet_cnn))


# In[131]:


# plotting the confusion matrix
cm = confusion_matrix(y_test,
                      y_pred_densenet_cnn)
sns.heatmap(cm,
            annot=True,
            cbar=False,
            fmt='g')
plt.title('DenseNet CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('plots/densenet_cnn_confusion_matrix.png',
            facecolor='white')
plt.show()


# In[132]:


# plotting scatter plot to visualize overlapping of predicted and
# test target data points
plt.scatter(range(len(y_pred_densenet_cnn)),
            y_pred_densenet_cnn,
            color='red')
plt.scatter(range(len(y_test)),
            y_test,
            color='green')
plt.title('DenseNet CNN - Predicted label vs. Actual label')
plt.xlabel('SMS Messages')
plt.ylabel('Label')
plt.savefig('plots/densenet_cnn_predicted_vs_real.png',
            facecolor='white')
plt.show()


# # Making Predictions with Real-World Examples
# 
# 

# In[133]:


# defining pre-processing hyperparameters

# maximum_length indicates the maximum number of words considered
# in a text
maximum_length = 50

# truncating_type indicates removal of values from sequences larger
# than maxlen, either at the beginning ('pre') or at the end ('post')
# of the sequences
truncating_type = 'post'

# padding_type indicates pad either before ('pre') or after ('post')
# each sequence
padding_type = 'post'


# In[134]:


# defining a function to preprocess the sms message text to feed to the
# trained deep learning models
# the input text is transformed to a sequence of integers
# then the sequence is padded to the same length
# padding='post' to pad after each sequence
# the function returns the padded sequence
def preprocess_text(sms_messages):
    sequence = tokenizer.texts_to_sequences(sms_messages)
    padded_sequence = pad_sequences(sequence,
                                    maxlen=maximum_length,
                                    padding=padding_type,
                                    truncating=truncating_type)
    return padded_sequence


# In[135]:


# defining a set of real-world samples for ham and spam sms messages
sms_messages = [
    'IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.',
    'Hello, Janith! Did you go to the school yesterday? If you did, can you please send me the notes of all the subjects?',
    'Congratulations ur awarded 500 of CD vouchers or 125 gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066.',
    'A loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.abc.co.uk to opt out reply stop',
    'If he started searching, he will get job in few days. He has great potential and talent.',
    'One chance ONLY! Had your mobile 11mths+? You are entitled to update to the latest colour camera mobile for FREE! Call The Mobile Update Co FREE on 08002986906.',
    'Valentines Day Special! Win over 1000 USD in cash in our quiz and take your partner on the trip of a lifetime! Send GO to 83600 now. 150 p/msg rcvd.',
    'Now I am better. Made up for Friday and stuffed myself like a pig yesterday. Now I feel bad.',
    'I got another job! The one at the hospital, doing data analysis or something, starts on Monday! Not sure when my thesis will finish.'
]


# In[136]:


# invoking the preprocess_text function to preprocess and get the
# padded sequences of the set of real-world samples for ham and
# spam sms messages
padded_sequences = preprocess_text(sms_messages)
padded_sequences


# In[137]:


# making prediction for the given set of real-world sms messages
# using the trained LSTM model

print('LSTM Model Predictions',
      end='\n\n')

lstm_prediction = []

for index, sms_message in enumerate(sms_messages):
    prediction = loaded_lstm_model.predict(padded_sequences)[index][0][0]
    lstm_prediction.append(prediction)
    if prediction > 0.5:
        print('SPAM -', prediction, '-', sms_message)
    else:
        print('HAM  -', prediction, '-', sms_message)


# In[138]:


# making prediction for the given set of real-world sms messages
# using the trained DenseNet CNN model

print('DenseNet CNN Model Predictions',
      end='\n\n')

densenet_cnn_prediction = []

for index, sms_message in enumerate(sms_messages):
    prediction = loaded_densenet_cnn_model.predict(padded_sequences)[index][0]
    densenet_cnn_prediction.append(prediction)
    if prediction > 0.5:
        print('SPAM -', prediction, '-', sms_message)
    else:
        print('HAM  -', prediction, '-', sms_message)


# # Comparison of Deep Learning Models

# In[139]:


# plotting line graphs of the predicted values for LSTM and
# DenseNet CNN deep learning models to compare the separation
# of classes by each model
plt.plot(list(range(len(sms_messages))),
         lstm_prediction,
         label='LSTM',
         color='blue',
         marker='o')
plt.plot(list(range(len(sms_messages))),
         densenet_cnn_prediction,
         label='CNN',
         color='red',
         marker='o')
plt.plot(list(range(len(sms_messages))),
         [0.5 for x in range(len(sms_messages))],
         color='black',
         linestyle='dashed')
plt.title('Comparison of Predicted Values by LSTM and DenseNet Models')
plt.xlabel('SMS Message ID')
plt.ylabel('Predicted Value')
plt.legend(loc='upper right',
           bbox_to_anchor=(1, 1))
plt.savefig('plots/prediction_values_comparison.png',
            facecolor='white')
plt.show()


# In[140]:


# importing recall_score from scikit-learn library
from sklearn.metrics import recall_score

# importing precision_score from scikit-learn library
from sklearn.metrics import precision_score

# importing f1_score from scikit-learn library
from sklearn.metrics import f1_score

accuracy = {}
recall = {}
precision = {}
f1 = {}
rmse = {}
mae = {}

y_pred_dict = {
    'LSTM': y_pred_lstm,
    'CNN': y_pred_densenet_cnn
}

for y_pred in y_pred_dict:
    accuracy[y_pred] = accuracy_score(y_test,
                                      y_pred_dict[y_pred])
    recall[y_pred] = recall_score(y_test,
                                  y_pred_dict[y_pred],
                                  average='weighted')
    precision[y_pred] = precision_score(y_test,
                                        y_pred_dict[y_pred],
                                        average='weighted')
    f1[y_pred] = f1_score(y_test,
                          y_pred_dict[y_pred],
                          average='weighted')
    rmse[y_pred] = mean_squared_error(y_test,
                                      y_pred_dict[y_pred],
                                      squared=False)
    mae[y_pred] = mean_absolute_error(y_test,
                                      y_pred_dict[y_pred])

# ratio of the number of correct predictions to the total number
# of input samples
print('Accuracy:', accuracy)

# recall is the ratio tp / (tp + fn)
# tp is the number of true positives
# fn the number of false negatives
print('Recall:', recall)

# precision is the ratio tp / (tp + fp)
# tp is the number of true positives
# fp the number of false positives
print('Precision:', precision)

# F1 score is also known as balanced F-score or F-measure
# F1 score can be interpreted as a weighted average of the
# precision and recall
# F1 = 2 * (precision * recall) / (precision + recall)
print('F1 Score:', f1)

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted
# and actual value of variable
print('RMSE:', rmse)

# mean absolute error (MAE)
print('MAE:', mae)


# In[141]:


# sorting the accuracy scores of two deep learning models in
# descending order
sorted(accuracy.items(),
       key=lambda kv: kv[1],
       reverse=True)


# In[142]:


# plotting the accuracy comparison bar chart for the two
# deep learning models
fig, ax = plt.subplots(figsize=(5, 5))
plt.bar(y_pred_dict.keys(),
        accuracy.values(),
        color='rg')
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.savefig('plots/accuracy_comparison.png',
            facecolor='white')
plt.show()


# In[143]:


# defining a function to plot a bar chart with multiple bars
def bar_plot(ax,
             data,
             colors=None,
             total_width=0.8,
             single_width=1,
             legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset,
                         y,
                         width=bar_width * single_width,
                         color=colors[i % len(colors)])
        bars.append(bar[0])
    if legend:
        ax.legend(bars,
                  data.keys())


# In[144]:


# plotting the algorithm comparison chart for all evaluation metrics

data = {}

for key in y_pred_dict.keys():
    data[key] = [accuracy[key],
                 recall[key],
                 precision[key],
                 f1[key],
                 rmse[key],
                 mae[key]]

fig, ax = plt.subplots(figsize=(7, 5))

bar_plot(ax,
         data,
         total_width=0.9,
         single_width=0.9)

plt.title('Algorithm Comparison')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.xticks(range(len(data[key])),
           ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'RMSE', 'MAE'])
plt.savefig('plots/algorithm_comparison.png',
            facecolor='white')
plt.show()


# In[144]:




