import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords
df = pd.read_csv("Student_Responces.csv")
df.head()
new_column_names = {
    'Gender':'gender',
    'Are you confident in your chosen career path?': 'confidence_in_career',
    'What factors affect your career decision-making process?': 'career_decision_factors',
    'How did you choose your career path?': 'career_choice_process',
    'Have you sought advice from a career counselor or career advisor?': 'career_advice_sought',
    'Did you face pressure in choosing a career path?': 'career_choice_pressure',
    'Do you think your current career choice aligns with your skills and strengths?': 'career_alignment',
    'Have you ever thought that the path you have chosen is not suitable for you? You could have done better if you had chosen another path.': 'doubts_about_career',
    'If you realize that your initially chosen career path is not suitable for you, are you confident in shifting to a different career path?': 'confidence_in_switching_career',
    'Do you believe that choosing the wrong career path can lead to academic failure?': 'career_path_and_academic_failure',
    'Do you think schools/colleges should offer career counseling and guidance programs?': 'school_career_counseling',
    'In your opinion, what can educational institutions do to help students choose their career paths more effectively? (Open-ended)': 'educational_institution_support'
}

df.rename(columns=new_column_names, inplace=True)
column_to_drop = ['educational_institution_support']
df.drop(column_to_drop, axis=1, inplace=True)

mapping = {
    'Parental expectations': 1,
    'Passion for a specific field': 2,
    'Financial stability': 3,
    'Job market trends': 4,
    'Other': 5
}

df['career_decision_factors'] = df['career_decision_factors'].replace(mapping)
mapping = {
    'Talked to friends/family': 1,
    'Based on financial stability': 2,
    'Through professional counseling': 3,
    'Based on personal interests and passion': 4,
    'Considering my educational background and qualifications': 5,
    'Other': 6
}

df['How_did you_choose_your_career_path? '] = df['How_did you_choose_your_career_path? '].replace(mapping)
mapping = {
    'gender': {'Male': 1, 'Female': 0},
    'confidence_in_career': {'Yes': 1, 'No': 0},
    'career_advice_sought': {'Yes': 1, 'No': 0},
    'career_choice_pressure': {'Yes': 1, 'No': 0},
    'career_alignment': {'Yes': 1, 'No': 0},
    'doubts_about_career': {'Yes': 1, 'No': 0},
    'confidence_in_switching_career': {'Yes': 1, 'No': 0},
    'career_path_and_academic_failure': {'Yes': 1, 'No': 0},
    'school_career_counseling': {'Yes': 1, 'No': 0},
   
}
df = df.replace(mapping)
df.columns
file_path = "processed_data.csv"
df.to_csv(file_path, index=False)
print("Data saved to", file_path)
columns_to_visualize = ['gender', 'confidence_in_career', 'career_decision_factors',
                             'How_did you_choose_your_career_path? ', 'career_advice_sought',
                             'career_choice_pressure', 'career_alignment', 'doubts_about_career',
                             'confidence_in_switching_career', 'career_path_and_academic_failure',
                             'school_career_counseling']
fig, axes = plt.subplots(4, 3, figsize=(15, 15))

axes = axes.flatten()

# Loop through the columns and create subplots
for i, column in enumerate(columns_to_visualize):
    sns.histplot(data=df[column], kde=True, ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(4, 3, figsize=(15, 15))

axes = axes.flatten()

# Loop through the columns and create subplots
for i, column in enumerate(columns_to_visualize):
    sns.histplot(data=df[column], kde=True, ax=axes[i])
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()
df['gender'].value_counts()
plt.pie(df['gender'].value_counts(),labels = ["Male","Female"],autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["Male","Female"])
plt.title('Proportion of gender')
plt.show()
df['confidence_in_career'].value_counts() 
plt.pie(df['confidence_in_career'].value_counts(),labels = ["yes","No"],autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["yes","No"])
plt.show()
df['career_decision_factors'].value_counts() 
plt.pie(df['career_decision_factors'].value_counts(),labels = ["Passion for a specific field","Financial stability","Job market trends","Parental expectations","Other"],autopct ="%.01f%%")
plt.show()
df['How_did you_choose_your_career_path? '].value_counts() 
plt.pie(df['How_did you_choose_your_career_path? '].value_counts(),labels = ["Based on personal interests and passion","Talked to friends/family","Based on financial stability","Considering my educational background and qualifications","Through professional counseling","Other"],autopct ="%.01f%%")
plt.show()
df['career_advice_sought'].value_counts() 
plt.pie(df['career_advice_sought'].value_counts(),labels = ["No","yes"],autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["No","yes"])
plt.show()
df['career_choice_pressure'].value_counts() 
plt.pie(df['career_choice_pressure'].value_counts(),labels = ["No","yes"],autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["No","yes"])
plt.show()
df['career_alignment'].value_counts() 
plt.pie(df['career_alignment'].value_counts(),labels = ["yes","No"] ,autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["yes","No"])
plt.show()
df['doubts_about_career'].value_counts() 
plt.pie(df['career_alignment'].value_counts(),labels = ["No","yes"] ,autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["No","yes"])
plt.show()
df['confidence_in_switching_career'].value_counts()
plt.pie(df['confidence_in_switching_career'].value_counts(),labels = ["No","yes"] ,autopct ="%.01f%%",explode=[0.1, 0],startangle=90)
plt.legend(["No","yes"])
plt.show()
df['career_path_and_academic_failure'].value_counts() 
plt.pie(df['career_path_and_academic_failure'].value_counts(),labels = ["yes","No"] ,autopct ="%.01f%%", explode=[0.1, 0],startangle=90)
plt.legend(["yes","No"])
plt.show()
df['school_career_counseling'].value_counts() 
plt.pie(df['school_career_counseling'].value_counts(),labels = ["yes","No"] ,autopct ="%.01f%%", explode=[0.1, 0],startangle=90)
plt.legend(["yes","No"])
plt.show()
df.columns
data =  """
 our school system is not perfect, and almost students do not know about what they can do in our practical life, they have no dreams. Therefore, we are not successful.

I think educational institutes must develop the students' minds in such a crystal & clear way that students do not feel shyness in choosing a career of their own interest. A teacher builds a Strong-Mind(Bull-Head). So, the role of the teacher is important to produce a "Strong and Healthy Mind Nation."

They should offer career counseling to students, make sessions

There should be the need for career counselors who timely motivate the students to choose the right path

N/A

My opinion is that there is a career counseling team made by the government in every college and university to give the true direction about the future of students.

Very good opinion

Every educational institution should learn the students that any career is not wrong; it's totally dependent upon you, how much your struggle involves in achieving that career.

Institutes should arrange career counseling sessions for students

Only teach Students and make it effective

Test their skills

A perfect career counseling session to let the students select their profession according to their interest.

Yes

Must provide career counseling and guidance. They must aware students of the best opportunities for their career according to their skills.

Encourage passions of students

There must be some practical activities for students. Based on the interest of students in these activities, they must select their field.

There should be proper career counseling of a student before choosing their academic career.

After getting the admissions, there should be counseling sessions so if anyone makes their mind so he should easily switch.

According to my opinion, educational institutions should organize career counseling seminars and sessions, then invite successful and most dedicated personalities of this era to motivate its students.

FIRST OF ALL, every student has its own willing and parental pressure. But there are no concepts of career counseling in Pakistan schools and colleges. Therefore, the student does not take great achievements and success; they fall into failure. That's why... my suggestion is that in every school and college, there must be extra lectures about career counseling. Then the student will decide the right path, what they can do in their future...

There should be a career counseling program in schools and colleges because students face difficulties in choosing the right profession.

Educational institutions should offer career counseling and guidance programs.

Students' interests matter. What kind of field they choose

Sorry 😐

In my opinion, teachers should also guide job opportunities in this field.

There should be a keen and thorough counseling from the school level. Teachers should identify one's skills and encourage them to pursue whatever a student wants besides any pressure.

They should motivate students & tell them about the opportunities they can get.

Career counseling is today's basic need for every student to choose the best career.

Yes

..

Check students' interest and guide them

I think we need to strengthen our school system. If our base was strong, it helps us in our practical life, and we will succeed.
"""
# Tokenize the text data
tokens = word_tokenize(data.lower())
# Remove stopwords (common words like 'the', 'and', 'is', etc.)
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
# Create a word frequency dictionary
word_freq = nltk.FreqDist(filtered_tokens)
 # Generate the Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to perform sentimental analysis
def perform_sentiment_analysis(text_data):
    sentences = nltk.sent_tokenize(text_data)
    sentiment_scores = {
        'Positive': 0,
        'Neutral': 0,
        'Negative': 0,
        'Compound': 0
    }
    for sentence in sentences:
        score = sia.polarity_scores(sentence)
        sentiment_scores['Positive'] += score['pos']
        sentiment_scores['Neutral'] += score['neu']
        sentiment_scores['Negative'] += score['neg']
        sentiment_scores['Compound'] += score['compound']
    total_sentences = len(sentences)
    for key in sentiment_scores:
        sentiment_scores[key] /= total_sentences
    return sentiment_scores
# Perform sentiment analysis on the text data
sentiment_scores = perform_sentiment_analysis(data)

# Display the results
print("Sentiment Scores:")
for key, value in sentiment_scores.items():
    print(f"{key}: {value:.3f}")

plt.show()
# Display the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.show()
