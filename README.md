# GenAI_Session2_hands_on
This repository contains an intermediate hands-on that can be used to understand the functioning of Basic NLP

### 1. Overview of NLP
**Example: Analyzing the Language in Shakespeare's Plays**

**Objective:** Introduce students to NLP by analyzing the language used in Shakespeare's plays.

**Activity:**
1. **Text Collection:**
   - Use the complete works of William Shakespeare as the dataset.
2. **Basic Analysis:**
   - Perform word frequency analysis.
   - Identify and visualize the most common words used by Shakespeare.

**Demonstration:**
```python
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

# Load Shakespeare's plays
shakespeare_texts = gutenberg.raw('shakespeare-hamlet.txt')

# Tokenize text
tokens = nltk.word_tokenize(shakespeare_texts)
tokens = [word.lower() for word in tokens if word.isalpha()]

# Analyze word frequency
word_freq = Counter(tokens)
common_words = word_freq.most_common(10)

# Plot the most common words
words, frequencies = zip(*common_words)
plt.bar(words, frequencies)
plt.title('Most Common Words in Hamlet by Shakespeare')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()
```

### 2. Generative Models in NLP
**Example: Generating New Lines for Famous Movie Characters**

**Objective:** Demonstrate the capabilities of generative models by creating new dialogues for famous movie characters.

**Activity:**
1. **Character Selection:**
   - Choose a famous movie character (e.g., Tony Stark from Iron Man).
2. **Dialogue Generation:**
   - Use a generative model to create new lines for the character.

**Demonstration:**
```python
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

# Generate new lines for Tony Stark
character = 'Tony Stark'
prompt = f"Write a new dialogue for {character} from Iron Man."
response = openai.Completion.create(
    engine="text-davinci-004",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7
)
print(f"{character}: {response.choices[0].text.strip()}")
```

### 3. Corpora, Tokens, and Features
**Example: Tokenizing and Analyzing the Harry Potter Series**

**Objective:** Teach students about corpora, tokens, and features by analyzing the Harry Potter books.

**Activity:**
1. **Text Collection:**
   - Use the text of the Harry Potter series.
2. **Tokenization:**
   - Tokenize the text into words and sentences.
3. **Feature Extraction:**
   - Extract features like named entities, noun phrases, etc.

**Demonstration:**
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load text from Harry Potter book
harry_potter_text = "Your Harry Potter text here"

# Tokenize text
sentences = nltk.sent_tokenize(harry_potter_text)
words = nltk.word_tokenize(harry_potter_text)

# Extract named entities
tags = nltk.pos_tag(words)
print(tags)
```

### 4. Text Preprocessing Techniques for Generative Models
**Example: Preprocessing the Text of "The Lord of the Rings"**

**Objective:** Show students how to preprocess text data for generative models using "The Lord of the Rings."

**Activity:**
1. **Text Cleaning:**
   - Remove special characters, numbers, and stopwords.
2. **Lemmatization and Stemming:**
   - Apply lemmatization and stemming to the text.

**Demonstration:**
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load text from The Lord of the Rings
lotr_text = "Your Lord of the Rings text here"

# Text cleaning
words = nltk.word_tokenize(lotr_text)
words = [word.lower() for word in words if word.isalpha()]
words = [word for word in words if word not in stopwords.words('english')]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print(lemmatized_words)
```

### 5. Introduction to Language Models for Generative AI
**Example: Using GPT-4 to Write an Alternative Ending to "The Great Gatsby"**

**Objective:** Introduce language models by generating an alternative ending for "The Great Gatsby."

**Activity:**
1. **Prompt Creation:**
   - Create a prompt that asks GPT-4 to write an alternative ending.
2. **Text Generation:**
   - Use GPT-4 to generate the new ending.

**Demonstration:**
```python
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

# Generate an alternative ending for The Great Gatsby
prompt = "Write an alternative ending for The Great Gatsby."
response = openai.Completion.create(
    engine="text-davinci-004",
    prompt=prompt,
    max_tokens=200,
    temperature=0.7
)
print("Alternative Ending:", response.choices[0].text.strip())
```

### 6. Comparison of GPT-3 and GPT-4o (and its enhancements)
**Example: Comparing Responses to Historical Event Prompts**

**Objective:** Compare GPT-3 and GPT-4o by generating responses to prompts about historical events.

**Activity:**
1. **Prompt Creation:**
   - Create a prompt related to a historical event (e.g., the Moon landing).
2. **Generate Responses:**
   - Use both GPT-3 and GPT-4o to generate responses.
3. **Comparison:**
   - Compare the quality, detail, and accuracy of the responses.

**Demonstration:**
```python
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

# Generate response using GPT-3
prompt = "Describe the significance of the Moon landing in 1969."
response_gpt3 = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=150,
    temperature=0.7
)
print("GPT-3 Response:", response_gpt3.choices[0].text.strip())

# Generate response using GPT-4o
response_gpt4 = openai.Completion.create(
    engine="text-davinci-004",
    prompt=prompt,
    max_tokens=150,
    temperature=0.7
)
print("GPT-4o Response:", response_gpt4.choices[0].text.strip())
```
