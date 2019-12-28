#Importing the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import string
import re
import warnings
warnings.filterwarnings("ignore")

#%%
def define_alphabet():
    base_en = ""
    for i in range(0,26):
        x = chr(97+i)
        base_en += str(x)
    special_chars = ' !?¿¡'
    german = 'äöüß'
    italian = 'àèéìíòóùú'
    french = 'àâæçéèêêîïôœùûüÿ'
    spanish = 'áéíóúüñ'
    czech = 'áčďéěíjňóřšťúůýž'
    slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
    all_lang_chars = base_en + german +  italian + french + spanish + czech + slovak
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort() 
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    letters_string = ''
    letters = small_chars + big_chars
    for letter in letters:
        letters_string += letter
    return small_chars,big_chars,letters_string

########
# Plot #
########

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        # heatmap for the dataframe of confusion matrix formed
        # annotation = True and the formatting inside cells set to integer
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

#%%
f=open('en.txt', 'r', encoding = 'utf-8').read(10000)

#%%
#Data Cleaning
#Making text all lower case
con = f.lower()
print(con)

#%%
#Removing XML tags
def remove_xml(text):
 return re.sub('<./>',"", text)

con = remove_xml(con)
print(con)

#%%
#Removing new lines to procure dense data
def remove_newlines(text):
  return text.replace("\n", " ")

con = remove_newlines(con)
print(con)

#%%
#Removing punctuations and numerical values
punc = string.punctuation
no_punc_num = " "
for ch in con:
  if ch not in punc and not ch.isdigit():
    no_punc_num = no_punc_num + ch
  
print(no_punc_num)

#%%
#Substituting many spaces with just one 
def remove_manyspaces(text):
  return re.sub(r'\s+', " ", text)

no_punc_num = remove_manyspaces(no_punc_num)
print(no_punc_num)

#%%
#Splitting by whitespace
no_punc_num = no_punc_num.split()
print(no_punc_num)

#%%
#Creating a WordCloud
wc_words = " "
stopwords = set(STOPWORDS)

for words in no_punc_num:
  wc_words = wc_words + words + " "

  wordcloud =  WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(wc_words) 
#Plotting the WordCloud image
  #%%
plt.figure(figsize = (15,15))
plt.imshow(wordcloud) 
plt.axis("off") 
plt.show()

#%%
def get_sample_text(file_content,start_index,sample_size):
    
    while not (file_content[start_index].isspace()):
        start_index += 1
    #now we look for first non-space character - beginning of any word
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index+sample_size 
    # we also want full words at the end
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]

def count_chars(text, alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts

def get_input_row(content,start_index,sample_size, alphabet):
    sample_text = get_sample_text(content,start_index,sample_size)
    counted_chars_all = count_chars(sample_text.lower(), alphabet[0])
    counted_chars_big = count_chars(sample_text, alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts