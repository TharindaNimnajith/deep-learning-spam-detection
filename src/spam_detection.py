#!/usr/bin/env python
# coding: utf-8

# In[1]:


# magic function that renders the figure in a notebook instead of
# displaying a dump of the figure object
# sets the backend of matplotlib to the 'inline' backend
# with this backend, the output of plotting commands is displayed inline
# within frontends like the Jupyter notebook, directly below the code cell
# that produced it
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


# importing warnings library to handle exceptions, errors, and warning of
# the program
import warnings

# ignoring potential warnings of the program
warnings.filterwarnings('ignore')


# In[4]:


# mounting google drive to read files stores in it
# from google.colab import drive

# drive.mount('/content/drive')


# In[5]:


get_ipython().system('wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip')


# In[6]:


get_ipython().system('unzip /content/smsspamcollection.zip')


# In[7]:


get_ipython().system('ls')


# In[8]:


# importing pandas library to perform data manipulation and analysis
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_colwidth = 150


# In[9]:


sms_spam_dataframe = pd.read_csv('/content/SMSSpamCollection',
                                 sep='\t',
                                 header=None,
                                 names=['class', 'sms_message'])
sms_spam_dataframe


# In[10]:


# printing the columns of the dataframe
sms_spam_dataframe.columns


# In[11]:


sms_spam_dataframe = sms_spam_dataframe[['sms_message', 'class']]
sms_spam_dataframe


# In[12]:


# displaying the dimensionality of the dataframe
sms_spam_dataframe.shape


# In[13]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values,
# and memory usage
sms_spam_dataframe.info()


# In[14]:


# generating descriptive statistics of the dataframe
sms_spam_dataframe.describe()


# In[15]:


sms_spam_dataframe.groupby('class').describe().T


# In[16]:


# checking for missing or null values in the dataframe
dataframe_null = sms_spam_dataframe[sms_spam_dataframe.isnull().any(axis=1)]
dataframe_null


# In[17]:


# number of rows with any missing or null values in the dataframe
dataframe_null.shape[0]


# In[18]:


# removing the missing or null values from the dataframe if exist
sms_spam_dataframe = sms_spam_dataframe[sms_spam_dataframe.notna().all(axis=1)]

# count of null values in class and sms_message columns of the dataframe
sms_spam_dataframe[['class', 'sms_message']].isnull().sum()


# In[19]:


# importing pyplot from matplotlib library to create interactive
# visualizations
import matplotlib.pyplot as plt


# In[20]:


# importing seaborn library which is built on top of matplotlib to create
# statistical graphics
import seaborn as sns


# In[21]:


# plotting the heatmap for missing or null values in the dataframe before
# cleaning
sns.heatmap(sms_spam_dataframe.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
plt.title('Null Values Detection Heat Map')
plt.savefig('plots/null_detection_heat_map.png',
            facecolor='white')
plt.show()


# In[22]:


# importing missingno library
# used to understand the distribution of missing values through
# informative visualizations
# visualizations can be in the form of heat maps or bar charts
# used to observe where the missing values have occurred
# used to check the correlation of the columns containing the missing
# with the target column
import missingno as msno

# plotting a matrix visualization of the nullity of the dataframe
# before cleaning
fig = msno.matrix(sms_spam_dataframe)
fig_copy = fig.get_figure()
fig_copy.savefig('plots/msno_matrix.png',
                 bbox_inches='tight')
fig


# In[23]:


# plotting a seaborn heatmap visualization of nullity correlation
# in the dataframe before cleaning
fig = msno.heatmap(sms_spam_dataframe)
fig_copy = fig.get_figure()
fig_copy.savefig('plots/msno_heatmap.png',
                 bbox_inches='tight')
fig


# In[24]:


duplicated_records = sms_spam_dataframe[sms_spam_dataframe.duplicated()]
duplicated_records


# In[25]:


# checking the number of duplicate rows exist in the dataframe
# before cleaning
sms_spam_dataframe.duplicated().sum()


# In[26]:


# removing the duplicate rows from the dataframe if exist
# sms_spam_dataframe = sms_spam_dataframe.drop_duplicates()


# In[27]:


# checking the number of duplicate rows exist in the dataframe
# after cleaning
# sms_spam_dataframe.duplicated().sum()


# In[28]:


# saving cleaned dataset to a csv file
file_name = 'processed_datasets/cleaned_dataset.csv'
sms_spam_dataframe.to_csv(file_name,
                          encoding='utf-8',
                          index=False)

# loading dataset from the saved csv file to a pandas dataframe
sms_spam_dataframe = pd.read_csv(file_name)

# printing the cleaned loaded dataframe
sms_spam_dataframe


# In[29]:


# displaying the first 5 rows of the dataframe
sms_spam_dataframe.head()


# In[30]:


# displaying the last 5 rows of the dataframe
sms_spam_dataframe.tail()


# In[31]:


# importing set of stopwords from wordcloud library
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

# printing number of stopwords defined in wordcloud library
len(stopwords)


# In[32]:


# importing WordCloud object for generating and drawing
# wordclouds from wordcloud library
from wordcloud import WordCloud


# In[33]:


# function to return the wordcloud for a given text
def plot_wordcloud(text):
    wordcloud = WordCloud(width=600,
                          height=300,
                          background_color='black',
                          stopwords=stopwords,
                          max_font_size=50,
                          colormap='Oranges').generate(text)
    return wordcloud


# In[34]:


sms_spam_dataframe['class'].value_counts()


# In[35]:


ham_dataframe = sms_spam_dataframe[sms_spam_dataframe['class'] == 'ham']
ham_dataframe


# In[36]:


# create numpy list to visualize using wordcloud
ham_sms_message_text = ' '.join(ham_dataframe['sms_message'].to_numpy().tolist())

# generate wordcloud for ham sms messages
ham_sms_wordcloud = plot_wordcloud(ham_sms_message_text)
plt.figure(figsize=(16, 10))
plt.imshow(ham_sms_wordcloud,
           interpolation='bilinear')
plt.axis('off')
plt.title('Ham SMS Wordcloud')
plt.savefig('plots/ham_wordcloud.png',
            facecolor='white')
plt.show()


# In[37]:


spam_dataframe = sms_spam_dataframe[sms_spam_dataframe['class'] == 'spam']
spam_dataframe


# In[38]:


# create numpy list to visualize using wordcloud
spam_sms_message_text = ' '.join(spam_dataframe['sms_message'].to_numpy().tolist())

# generate wordcloud for spam sms messages
spam_sms_wordcloud = plot_wordcloud(spam_sms_message_text)
plt.figure(figsize=(16, 10))
plt.imshow(spam_sms_wordcloud,
           interpolation='bilinear')
plt.axis('off')
plt.title('Spam SMS Wordcloud')
plt.savefig('plots/spam_wordcloud.png',
            facecolor='white')
plt.show()


# In[39]:


# plotting the distribution of target values
fig = plt.figure()
lbl = ['Ham (0)', 'Spam (1)']
pct = '%1.0f%%'
ax = sms_spam_dataframe['class'].value_counts().plot(kind='pie',
                                                     labels=lbl,
                                                     autopct=pct)
ax.yaxis.set_visible(False)
plt.title('Distribution of Ham and Spam SMS')
plt.legend()
fig.savefig('plots/ham_spam_pie_chart.png',
            facecolor='white')
plt.show()


# In[40]:


# downsampling is a process where you randomly delete some of the
# observations from the majority class so that the numbers in majority
# and minority classes are matched. Below, we have downsampled the ham
# messages (majority class)
# there are now 747 messages in each class
downsampled_ham_dataframe = ham_dataframe.sample(n=len(spam_dataframe),
                                                 random_state=44)
downsampled_ham_dataframe


# In[41]:


print('Ham dataframe shape:', downsampled_ham_dataframe.shape)
print('Spam dataframe shape:', spam_dataframe.shape)


# In[42]:


# mergin the two dataframes (spam + downsampled ham dataframes)
merged_dataframe = pd.concat([downsampled_ham_dataframe, spam_dataframe])
merged_dataframe = merged_dataframe.reset_index(drop=True)
merged_dataframe 


# In[43]:


merged_dataframe['class'].value_counts()


# In[44]:


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


# In[45]:


merged_dataframe['label'] = merged_dataframe['class'].map({'ham': 0, 'spam': 1})
merged_dataframe


# In[46]:


merged_dataframe['length'] = merged_dataframe['sms_message'].apply(len)
merged_dataframe


# In[47]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values,
# and memory usage
merged_dataframe.info()


# In[48]:


# generating descriptive statistics of the dataframe
merged_dataframe.describe().round(2)


# In[49]:


# calculate length statistics by label types
merged_dataframe.groupby('class')['length'].describe().round(2)


# In[50]:


# plot a univariate distribution of observations for sms lengths
sns.distplot(merged_dataframe['length'].values)
plt.title('SMS Lengths Distribution')
plt.xlabel('SMS Length')
plt.savefig('plots/sms_length.png',
            facecolor='white')
plt.show()


# In[51]:


# saving merged dataset to a csv file
file_name = 'processed_datasets/merged_dataset.csv'
merged_dataframe.to_csv(file_name,
                        encoding='utf-8',
                        index=False)

# loading dataset from the saved csv file to a pandas dataframe
merged_dataframe = pd.read_csv(file_name)

# printing the cleaned loaded dataframe
merged_dataframe


# In[52]:


X = merged_dataframe['sms_message']
X


# In[53]:


y = merged_dataframe['label'].values
y


# In[54]:


# importing train_test_split from scikit-learn library
from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=443)


# In[56]:


X_train


# In[57]:


X_train.shape


# In[58]:


y_train


# In[59]:


y_train.shape


# In[60]:


X_test


# In[61]:


X_test.shape


# In[62]:


y_test


# In[63]:


y_test.shape


# In[64]:


# importing tensorflow library
# it is a free and open-source software library for machine learning
# used across a range of machine learning related tasks
# focus on training and inference of deep neural networks
import tensorflow as tf


# In[65]:


# importing Tokenizer from keras library
# keras is a high-level api of tensorflow
# keras.preprocessing.text provides keras data preprocessing utils
# to pre-process datasets with textual data before they are fed to the
# machine learning model
from tensorflow.keras.preprocessing.text import Tokenizer


# In[66]:


# defining pre-processing hyperparameters

# out of vocabulary token
oov_token = '<OOV>'

vocabulary_size = 500


# In[67]:


# Tokenizer allows to vectorize a text corpus, by turning each text into
# either a sequence of integers (each integer being the index of a token
# in a dictionary) or into a vector where the coefficient for each token
# could be binary, based on word count, based on tf-idf

# hyper-parameters used in Tokenizer object are: num_words and oov_token
# num_words: indicate how many unique word you want to load in training
# and testing data
# oov_token: out of vocabulary token will be added to word index in the
# corpus which is used to build the model. This is used to replace out of
# vocabulary words (words that are not in our corpus) during
# text_to_sequence calls.

tokenizer = Tokenizer(num_words=vocabulary_size,
                      char_level=False,
                      oov_token=oov_token)

# updates internal vocabulary based on a list of texts
# required before using texts_to_sequences
tokenizer.fit_on_texts(X_train)


# In[68]:


# get the word_index
word_index = tokenizer.word_index
word_index


# In[69]:


len(word_index)


# In[70]:


# defining pre-processing hyperparameters

# idicates the maximum number of words considered in a text
maximum_length = 50

# remove values from sequences larger than maxlen, either at the
# beginning ('pre') or at the end ('post') of the sequences
truncating_type = 'post'

# pad either before ('pre') or after ('post') each sequence
padding_type = 'post'


# In[71]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[72]:


# importing numpy library
# used to perform fast mathematical operations over python arrays
# and lists
import numpy as np


# In[73]:


# transforms each text in train data to a sequence of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_sequences[0]


# In[74]:


# lengths of each generated sequences of integers in train data
x_train_length_of_sequence = [len(sequence) for sequence in X_train_sequences]
x_train_length_of_sequence[0]


# In[75]:


# maximum length of a sequence in the train data
np.max(x_train_length_of_sequence)


# In[76]:


# plot a univariate distribution of observations for sequence lengths
# of train data
sns.distplot(x_train_length_of_sequence)
plt.title('Train Data Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/train_sequence_length.png',
            facecolor='white')
plt.show()


# In[77]:


# transforms each text in test data to a sequence of integers
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_sequences[0]


# In[78]:


# lengths of each generated sequences of integers in test data
x_test_length_of_sequence = [len(sequence) for sequence in X_test_sequences]
x_test_length_of_sequence[0]


# In[79]:


# maximum length of a sequence in the test data
np.max(x_test_length_of_sequence)


# In[80]:


# plot a univariate distribution of observations for sequence lengths
# of test data
sns.distplot(x_test_length_of_sequence)
plt.title('Test Data Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/test_sequence_length.png',
            facecolor='white')
plt.show()


# In[81]:


# padding on train data
X_train_padded = pad_sequences(X_train_sequences,
                               maxlen=maximum_length,
                               padding=padding_type,
                               truncating=truncating_type)
X_train_padded[0]


# In[82]:


# lengths of each padded sequences of integers in train data
x_train_length_of_padded_sequence = [len(sequence) for sequence in X_train_padded]
x_train_length_of_padded_sequence[0]


# In[83]:


# maximum length of a padded sequence in the train data
np.max(x_train_length_of_padded_sequence)


# In[84]:


X_train_padded.shape


# In[85]:


# plot a univariate distribution of observations for sequence lengths
# of train data after padding
sns.distplot(x_train_length_of_padded_sequence)
plt.title('Train Data Padded Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/train_padded_sequence_length.png',
            facecolor='white')
plt.show()


# In[86]:


# padding on test data
X_test_padded = pad_sequences(X_test_sequences,
                              maxlen=maximum_length,
                              padding=padding_type,
                              truncating=truncating_type)
X_test_padded[0]


# In[87]:


# lengths of each padded sequences of integers in test data
x_test_length_of_padded_sequence = [len(sequence) for sequence in X_test_padded]
x_test_length_of_padded_sequence[0]


# In[88]:


# maximum length of a padded sequence in the test data
np.max(x_test_length_of_padded_sequence)


# In[89]:


X_test_padded.shape


# In[90]:


# plot a univariate distribution of observations for sequence lengths
# of test data after padding
sns.distplot(x_test_length_of_padded_sequence)
plt.title('Test Data Padded Sequence Lengths Distribution')
plt.xlabel('Sequence Length')
plt.savefig('plots/test_padded_sequence_length.png',
            facecolor='white')
plt.show()


# In[91]:


# LSTM network arcitecture hyperparameters

# SpatialDropout1D is used to dropout the embedding layer
# The SpatialDropout1D helps to drop entire 1D feature maps instead of
# individual elements.
dropout_rate = 0.2

# n_lstm is the number of nodes in the hidden layers within the LSTM cell
no_of_nodes = 20

# return_sequences=True ensures that the LSTM cell returns all of the
# outputs from the unrolled LSTM cell through time. If this argument is
# not used, the LSTM cell will simply provide the output of the LSTM cell
# from the previous step.

# 
embedding_dimension = 16

# 
no_of_epochs = 30

# 
vocabulary_size = 500


# In[92]:


# 
from tensorflow.keras.models import Sequential

# 
from tensorflow.keras.layers import Embedding

# 
from tensorflow.keras.layers import LSTM

# 
from tensorflow.keras.layers import Dense


# In[93]:


# LSTM model architecture

lstm_model = Sequential()

lstm_model.add(Embedding(vocabulary_size,
                         embedding_dimension,
                         input_length=maximum_length))

lstm_model.add(LSTM(no_of_nodes,
                    dropout=dropout_rate,
                    return_sequences=True))

lstm_model.add(LSTM(no_of_nodes,
                    dropout=dropout_rate,
                    return_sequences=True))

lstm_model.add(Dense(1,
                     activation='sigmoid'))


# In[94]:


lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])


# In[95]:


# printing a string summary of the network
lstm_model.summary()


# In[96]:


# 
from tensorflow.keras.callbacks import EarlyStopping


# In[97]:


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5)

history = lstm_model.fit(X_train_padded,
                         y_train,
                         epochs=no_of_epochs,
                         validation_data=(X_test_padded, y_test),
                         callbacks=[early_stopping],
                         verbose=2)


# In[98]:


lstm_path = 'models/lstm_model.h5'
lstm_model.save(lstm_path)


# In[99]:


from tensorflow.keras.models import load_model

loaded_lstm_model = load_model(lstm_path)


# In[100]:


loaded_lstm_model.evaluate(X_test_padded,
                           y_test)


# In[101]:


def preprocess_text(sms_messages):
    sequence_ = tokenizer.texts_to_sequences(sms_messages)
    padded_sequence = pad_sequences(sequence_,
                                    maxlen=maximum_length,
                                    padding=padding_type,
                                    truncating=truncating_type)
    return padded_sequence


# In[102]:


sms_messages = ['IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.',
                'Hello, Janith! Did you go to the school yesterday? If you did, can you please send me the notes of all the subjects?',
                'Congratulations ur awarded 500 of CD vouchers or 125 gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066.',
                'A loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.abc.co.uk to opt out reply stop',
                'If he started searching, he will get job in few days. He has great potential and talent.',
                'One chance ONLY! Had your mobile 11mths+? You are entitled to update to the latest colour camera mobile for FREEE! Call The Mobile Update Co FREE on 08002986906.',
                'Valentines Day Special! Win over 1000 USD in cash in our quiz and take your partner on the trip of a lifetime! Send GO to 83600 now. 150 p/msg rcvd.',
                'Now I am better. Made up for Friday and stuffed myself like a pig yesterday. Now I feel bad.',
                'I got another job! The one at the hospital, doing data analysis or something, starts on Monday! Not sure when my thesis will finish.']


# In[103]:


padded_sequences = preprocess_text(sms_messages)

for index, sms_message in enumerate(sms_messages):
  prediction = loaded_lstm_model.predict(padded_sequences)[index][0][0]
  if prediction > 0.5:
    print('SPAM - ', prediction, '-', sms_message)
  else:
    print('HAM  - ', prediction, '-', sms_message)


# In[104]:


# Densely Connected CNN (DenseNet) arcitecture hyperparameters

# 
dropout_rate = 0.2

# 
embedding_dimension = 16

# 
no_of_epochs = 30

# 
vocabulary_size = 500


# In[105]:


# 
from tensorflow.keras.models import Sequential

# 
from tensorflow.keras.layers import Embedding

# 
from tensorflow.keras.layers import GlobalAveragePooling1D

# 
from tensorflow.keras.layers import Dropout

# 
from tensorflow.keras.layers import Dense


# In[106]:


# Densely Connected CNN (DenseNet) model architecture

densenet_cnn_model = Sequential()

densenet_cnn_model.add(Embedding(vocabulary_size,
                                 embedding_dimension,
                                 input_length=maximum_length))

densenet_cnn_model.add(GlobalAveragePooling1D())

densenet_cnn_model.add(Dense(24,
                             activation='relu'))

densenet_cnn_model.add(Dropout(dropout_rate))

densenet_cnn_model.add(Dense(1,
                             activation='sigmoid'))


# In[107]:


densenet_cnn_model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


# In[108]:


# printing a string summary of the network
densenet_cnn_model.summary()


# In[109]:


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5)

history = densenet_cnn_model.fit(X_train_padded,
                                 y_train,
                                 epochs=no_of_epochs,
                                 validation_data=(X_test_padded, y_test),
                                 callbacks=[early_stopping],
                                 verbose=2)


# In[110]:


densenet_cnn_path = 'models/densenet_cnn_model.h5'
densenet_cnn_model.save(densenet_cnn_path)


# In[111]:


loaded_densenet_cnn_model = load_model(densenet_cnn_path)


# In[112]:


loaded_densenet_cnn_model.evaluate(X_test_padded,
                                   y_test)


# In[113]:


padded_sequences = preprocess_text(sms_messages)

for index, sms_message in enumerate(sms_messages):
  prediction = loaded_densenet_cnn_model.predict(padded_sequences)[index][0]
  if prediction > 0.5:
    print('SPAM - ', prediction, '-', sms_message)
  else:
    print('HAM  - ', prediction, '-', sms_message)


# In[113]:



