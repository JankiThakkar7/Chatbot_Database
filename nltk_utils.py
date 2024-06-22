import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import mysql.connector
from mysql.connector import Error

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

def load_data_from_database():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rcc_chatbot',
            port=3306,
            auth_plugin='mysql_native_password'
        )
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT d.id, d.category_id, d.patterns, d.responses, d.y_tags, c.category_name
        FROM dataset d
        JOIN categories c ON d.category_id = c.category_id
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        data = []
        for row in rows:
            category_id = row['category_id']
            category_name = row['category_name']
            patterns = row['patterns']
            responses = row['responses']
            y_tags = row['y_tags']  # Expecting single tag here

            # Split patterns for all categories
            patterns_list = patterns.split('|')

            # Split responses for category 1 and keep as a single string for category 2
            if category_id == 1:
                responses_list = responses.split('|')
            elif category_id == 2:
                responses_list = [responses]

            # Use the single tag
            tag = y_tags

            for pattern in patterns_list:
                data.append({
                    'category': category_name,
                    'pattern': pattern,
                    'responses': responses_list,
                    'tag': tag
                })

        conn.close()
        
        return data

    except mysql.connector.Error as e:
        print(f"Error fetching data from MySQL: {e}")
        return None
