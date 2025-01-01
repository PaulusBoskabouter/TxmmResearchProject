import os
import json
import threading
import os
import spacy
import nltk
import gensim
from nltk.corpus import stopwords
import numpy as np


class Thread(threading.Thread):
    def __init__(self, i, start_year, end_year):
        super().__init__()
        self.id = i
        self.start_year = start_year
        self.stop_year = end_year
        self.year_data = {}
    

    def preprocess(self, text:str) -> list[str]:
        text = text.lower()
        doc = nlp(text)
        
        # 4. Lemmatization
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

        return lemmatized_tokens



    # Create document vectors
    def get_document_vector(self, doc, model):
        """Compute the document vector as the average of the word vectors."""
        valid_words = [word for word in doc if word in model.key_to_index]
        if not valid_words:  # If no valid words, return a zero vector
            return None, True
        
        vector = np.array([model[word] for word in valid_words])
        
        word_vectors = np.mean(vector, axis=0)
        print(vector)
        print(word_vectors)
        print(vector.shape)
        print(word_vectors.shape)
        quit()
        return word_vectors.tolist(), False

    def run(self):
        self.iterate_files()


    def iterate_files(self):
        for year in range(self.start_year, self.stop_year+1):
            if not os.path.isfile(f"{year}.json"):
                print(f"{self.id:2}: Working on year {year} ({self.start_year}-{self.stop_year})")
                paragraph_data = {}
                year = str(year)
                files = os.listdir(os.path.join(root_path, str(year)))
                for f in range(len(files)):
                    # Visualisation stuff
                    file = files[f]
                    preprocessed_paragraphs = []
                    document_vectors = []
                    s = file.split('_')
                    file_name = s[1].lower()
                    idx = s[0]
                    used_articles = 0
                    with open(os.path.join(root_path, year, file), 'r', encoding='utf-8') as f:
                        text = f.read().lower()  
                        paragraphs = text.split("---")[:-1] # Last "paragraph" always consists of just \n so we remove this
                        for article in paragraphs:
                            preprocessed = self.preprocess(article)
                            vector, filtered = self.get_document_vector(preprocessed, fasttext_model)
                            if not filtered:
                                preprocessed_paragraphs.append(preprocessed)
                                document_vectors.append(vector)
                                used_articles +=1
                    paragraph_data[idx] = {'name':file_name, 'num_articles':len(paragraphs),'used_articles':used_articles, 'paragraphs':preprocessed_paragraphs, 'embedded':document_vectors}
                self.save(filename=f"{year}.json", data=paragraph_data)
                paragraph_data.clear()
            else:
                print(f"{self.id:2}: Skipping year {year} ({self.start_year}-{self.stop_year})")
        
        
        
        
    def save(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
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
    start_year = 1850
    end_year = 1995
    num_threads = 29
    years_per_thread = 5  # Base number of years per thread
    

    # Generate ranges
    thread_ranges = generate_ranges(start_year, years_per_thread, num_threads, end_year)
    threads = []

    # Display ranges
    for thread_id, (start, end) in enumerate(thread_ranges, 1):
        threads.append(Thread(thread_id, start, end))



    for t in threads:
        t.start()

    for t in threads:
        t.join()


    
            

