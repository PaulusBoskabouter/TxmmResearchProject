import os
import json
import threading
import os
import spacy
import nltk
import gensim
from nltk.corpus import stopwords
import numpy as np
import re

class Thread(threading.Thread):
    def __init__(self, i, start_year, end_year):
        super().__init__()
        self.id = i
        self.start_year = start_year
        self.stop_year = end_year
        self.year_data = {}
    
    
    def preprocess_text(self, text:str) -> list[str]:
        """"
        Preprocess the text by removing punctuation, numbers, stopwords, and lemmatizing the words.
        """

        clean_text = text.replace('\n', ' ')            # Remove newline characters
        
        # 1. Clean-up the text a bit
        clean_text = re.sub(r'[^\w\s]', '', clean_text) # Remove punctuation
        clean_text = re.sub(r'\d+', '', clean_text)     # Remove numbers
        clean_text = clean_text.lower()                 # Convert to lowercase

        # 2. Processing the text with spaCy
        doc = nlp(clean_text)

        # 3. Removing punctuation and non-alphabetic tokens
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

        return cleaned_tokens



    # Create document vectors
    def get_document_vector(self, doc, model) -> tuple[list, bool]:
        """
        Compute the document vector as the average of the word vectors.
        """
        valid_words = [word for word in doc if word in model.key_to_index]
        if not valid_words:  # If no valid words, we filter it out be returning True
            return None, True
        
        vector = np.array([model[word] for word in valid_words])
        word_vectors = np.mean(vector, axis=0)
        return word_vectors.tolist(), False

    def run(self):
        """"
        Main thread function that processes the files for the given year range.
        """	

        self.iterate_files()
        print(f"{self.id:2}: Finished processing years {self.start_year}-{self.stop_year}")


    def iterate_files(self):
        """
        The main loop that iterates over the files for the given year range.
        """
        for year in range(self.start_year, self.stop_year+1):
            if not os.path.isfile(f"{year}.json"):
                print(f"{self.id:2}: Working on year {year} ({self.start_year}-{self.stop_year})")
                year_data = {}
                year = str(year)
                files = os.listdir(os.path.join(root_path, str(year)))
                paper_names = []
                used_articles = 0
                total_articles = 0
                preprocessed_paragraphs = []
                document_vectors = []
                for file in files:
                    # Visualisation stuff
                    s = file.split('_')
                    file_name = s[1].lower()
                    with open(os.path.join(root_path, year, file), 'r', encoding='utf-8') as f:
                        text = f.read().lower()  
                        paragraphs = text.split("---")[:-1] # Last "paragraph" always consists of just \n so we remove this
                        total_articles += len(paragraphs)
                        for article in paragraphs:
                            lemmatized = self.preprocess_text(article)
                            if len(lemmatized) > 400: # articles should consist of at least 100 words
                                vector, filtered = self.get_document_vector(lemmatized, fasttext_model)
                                if not filtered:
                                    paper_names.append(file_name)
                                    preprocessed_paragraphs.append(lemmatized)
                                    document_vectors.append(vector)
                                    used_articles +=1
                
                year_data = {'articles':(paper_names), 'total_articles':total_articles, 'used_articles':used_articles, 'paragraphs':preprocessed_paragraphs, 'embedded':document_vectors}
                self.save(filename=f"{year}.json", data=year_data)
            else:
                print(f"{self.id:2}: Skipping year {year} ({self.start_year}-{self.stop_year})")
        
        
        
        
    def save(self, filename, data):
        """ 
        Save the data to a JSON file. 
        """

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=1)
            file.close()


        
    
    



def generate_ranges(start_year, years_per_thread, num_threads, end_year):
    """
    Generate fixed year ranges for threads, ensuring the last range ends at the specified year.

    Parameters:
        start_year (int): The starting year of the range.
        years_per_thread (int): Base number of years each thread handles.
        num_threads (int): Total number of threads.
        end_year (int): The year the last range should end.

    Returns:
        list of tuples: Each tuple contains the start and end year for a thread.
    """
    ranges = []
    current_start = start_year
    total_years = end_year - start_year + 1
    extra_years = total_years % num_threads  # Remaining years to distribute

    for i in range(num_threads):
        # Distribute an extra year to the first few threads if there's a remainder
        extra = 1 if i < extra_years else 0
        range_end = current_start + years_per_thread + extra - 1
        ranges.append((current_start, range_end))
        current_start = range_end + 1

    return ranges





if __name__ == "__main__":
    # Download necessary NLTK resources
    nltk.download('stopwords')  # For stopwords

    embedding_file = "cc.nl.300.vec"  # Pre-trained Dutch embeddings file
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

    nlp = spacy.load("nl_core_news_sm")
    root_path = ".\\dataset\\"
    stop_words = set(stopwords.words('dutch'))
    

    # Parameters
    start_year = 1855
    end_year = 1995
    num_threads = 20
    years_per_thread = 7  # Base number of years per thread
    

    # Generate ranges
    thread_ranges = generate_ranges(start_year, years_per_thread, num_threads, end_year)
    threads = []

    # Create the threads
    for thread_id, (start, end) in enumerate(thread_ranges, 1):
        threads.append(Thread(thread_id, start, end))

    # Start the threads
    for t in threads:
        t.start()

    for t in threads:
        t.join()
