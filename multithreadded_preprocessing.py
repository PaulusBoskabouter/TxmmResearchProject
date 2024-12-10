import os
from time import sleep, time
from spellchecker import SpellChecker
import threading
import statistics

class Thread(threading.Thread):
    def __init__(self, i, start_year, end_year):
        super().__init__()
        self.id = i
        self.start_year = start_year
        self.stop_year = end_year
        self.domain = {}
        
        
    def run(self):
        print(f"T{self.id} is Starting! ({self.start_year}-{self.stop_year})")
        self.iterate_files()

    def iterate_files(self):
        root_path = ".\\dataset\\"
        processed_path = ".\\preprocessed\\"
        
        for year in range(self.start_year, self.stop_year):
            year = str(year)
            
            files = os.listdir(os.path.join(root_path, year))
            
            
            num_fixed_files = 0
            total_files = len(files)
            start = time()
            for file in files:
                with open(os.path.join(root_path, year, file), 'r', encoding='utf-8') as f:
                    text = f.read()  
                    if filter_spelling_mistakes(text, file_loc=os.path.join(processed_path, year), file_name=file):
                        num_fixed_files += 1
                    f.close()

            
            print(f"{year}: {num_fixed_files} out of {total_files} in {(time()-start)/60:.1f} Minutes")
        
        
        
    
    



def filter_spelling_mistakes(text, file_loc, file_name, language='nl'):
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
    

    corrected = {}
    


    for i in range(len(misspelled)):
        start = time()
        spell_correction = spell.correction(misspelled[i])
        change = False
        if misspelled[i][:-1].lower() != spell_correction: # If not this then words with interpunction get corrected.
            
            if spell_correction is None: # No correction could be found, we omit the word.
                spell_correction = "_"
            corrected[misspelled[i]] = spell_correction 
            change = True

        if change:
            print(f"Fixed {misspelled[i]} into {spell_correction} ({time()-start})\t[{i}/{len(misspelled)}]")
        else:
            print(f"Fixed {misspelled[i]} into itself ({time()-start})\t[{i}/{len(misspelled)}]")





    if mispelled_ratio < .25:
        fixed = ""
        os.makedirs(file_loc, exist_ok=True)
        for w in range(len(words)):
            start = time()
            word = words[w]

            
            if corrected.get(word) is not None:
                word = corrected.get(word)
        
            fixed += word + " "
            print(f"Written a word {w}/{len(words)} ({time()-start})")
        with open(os.path.join(file_loc, file_name), "w+", encoding="utf-8") as file:
            file.write(fixed)
        # We have fixed the file
        return True
    
    # Don't fix file
    return False


def generate_adjusted_ranges(start_year, years_per_thread, num_threads, end_year):
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

# Parameters
start_year = 1850
end_year = 1995
num_threads = 1
years_per_thread = 5  # Base number of years per thread

# Generate ranges
thread_ranges = generate_adjusted_ranges(start_year, years_per_thread, num_threads, end_year)
threads = []

# Display ranges
for thread_id, (start, end) in enumerate(thread_ranges, 1):
    threads.append(Thread(thread_id, start, end))



for t in threads:
    t.start()

for t in threads:
    t.join()