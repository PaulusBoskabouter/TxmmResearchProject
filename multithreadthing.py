import os
from time import sleep, time
from spellchecker import SpellChecker
import threading


class Thread(threading.Thread):
    def __init__(self, i, start_year, end_year):
        super().__init__()
        self.id = i
        self.start_year = start_year
        self.stop_year = end_year + 1
        self.domain = {}
        
        
    def run(self):
        self.iterate_files()
        self.log()

    def iterate_files(self):
        root_path = ".\\dataset\\"
        #root = os.listdir(root_path)

        for year in range(self.start_year, self.stop_year):
            year = str(year)
            delete = 0
            for file in os.listdir(os.path.join(root_path, year)):
                with open(os.path.join(root_path, year, file), 'r', encoding='utf-8') as f:
                    text = f.read()  
                    
                    if filter_spelling_mistakes(text):
                        delete += 1
                    f.close()
        
            self.domain[year] = delete 
            print(f"{year} should remove: {delete}/{len(os.listdir(os.path.join(root_path, year)))}")
    
    def log(self):
        with open(f"{self.id}_log.txt", "w+") as file:
            for y, d in self.domain.items():
                file.write(f"{str(y)}\t{str(d)}\n")






def filter_spelling_mistakes(text, language='nl'):
    """
    Identify spelling mistakes in the text based on the specified language.
    
    Parameters:
        text (str): Input text to analyze.
        language (str): The language for spell checking (default is 'nl' for Dutch).
        
    Returns:
        list: A list of words identified as spelling mistakes.
    """
    # Initialize the spell checker for the given language
    spell = SpellChecker(language=language)

    # Split text into words
    words = text.split()
    # Find words not in the dictionary
    misspelled = list(spell.unknown(words))

    mispelled_ratio = round(len(misspelled)/len(words), 2)

    corrected = []

    # TODO init fixing mistakes
    # for mispoes in misspelled:
    #     print(mispoes, spell.correction(mispoes))
    
    
    if mispelled_ratio > .25:
        return True
    
    return False






t1 = Thread(1, 1850, 1870)
t2 = Thread(2, 1871, 1890)
t3 = Thread(3, 1891, 1910)
t4 = Thread(4, 1911, 1930)
t5 = Thread(5, 1931, 1950)
t6 = Thread(6, 1951, 1970)
t7 = Thread(7, 1971, 1990)
t8 = Thread(8, 1991, 1995)


t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
    
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()            
t7.join()
t8.join() 