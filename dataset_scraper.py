import requests
import os
import regex as re
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
from time import sleep, time
from random import uniform


def extract_html(driver:webdriver.Firefox, year:int, page_number:int=1)-> str:
    """
    Given some year and month; open the driver and fetch the page source code 
    """
    base_link = f"https://www.delpher.nl/nl/kranten/results?query=&facets%5Bspatial%5D%5B%5D=Landelijk&page={page_number}&maxperpage=50&cql%5B%5D=(date+_gte_+%2201-01-{year}%22)&cql%5B%5D=(date+_lte_+%2231-12-{year}%22)&coll=dddtitel"
    
    
    # Open the page
    driver.get(base_link)

    
    #sleep(8) # Results are loaded in via JavaScript so we need to wait + it's nice for their server to not have me spam it.
    html = driver.page_source
    attempt = 1
    max_attempt = 10
    while not re.findall(r"([0-9]{1,2}-[0-9]{1,2}-[0-9]{4}).+?<\/a>", html):
        if attempt > max_attempt:
            print(f"No results found".ljust(150), end="\r")
            sleep(2)
            return html, True
        sleep(2)
        html = driver.page_source
        print(f"Awaiting results [{attempt}/{max_attempt}]".ljust(150), end="\r")
        attempt += 1
    else:
        print(f"Results found!".ljust(150), end="\r")


    return html, False


def extract_titles_codes(html:str, year:str) -> tuple:
    """
    We extract dates and codes from the source html file. 
    """
    code_and_title_re = r"<a href=\".+?\" title=\".+?\" class=\"search-result__link\">"
    
    
    
    extracted_dates = re.findall(r"([0-9]{1,2}-[0-9]{1,2}-[0-9]{4}).+?<\/a>", html)
    extracted_titles = []
    extracted_codes  = []
    
    results = re.findall(code_and_title_re, html)
    for c in results:
        # Extract title without "title=" and quotes
        title = re.search(r'title="([^"]+)"', c).group(1) # Output example: Nederlandsche staatscourant
        extracted_titles.append(title)

        # Extract identifier without "identifier="
        code = re.search(r'identifier=([^&]+)', c).group(1) # Output example: ddd:010783315:mpeg21
        extracted_codes.append(code)

    titles = []
    codes = []
    dates = []


    for i in range(len(extracted_dates)):
        if extracted_dates[i].split("-")[2] == str(year):
            titles.append(extracted_titles[i])
            codes.append(extracted_codes[i])
            dates.append(extracted_dates[i])



    return titles, codes, dates


def download_content(titles:list, codes:list, dates:list, year:int, target_downloads:int = 101):
    """
    We extract dates and codes from the source html file. 
    """

    assert len(titles) == len(codes) and len(dates) == len(codes) and len(titles) == len(dates), f"titles, codes and dates extraction went wrong for {year}"
   
    folder_path = f'dataset\\{year}'
    os.makedirs(folder_path, exist_ok=True)

    downloads = 1
    number_of_downloads = len(titles)

    tries = 1
    try_limit = 5

    while codes and downloads < target_downloads:
        wait_time = round(uniform(0.5, 1), 3)
        actually_written = False

        code = codes[0]
        title = titles[0]
        date = dates[0]
        


        download_string = f"https://www.delpher.nl/nl/pres/view/pageocr?identifier={code}&coll=ddd&operation=download"

        # Define the custom filename
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        code = re.sub(r'[<>:"/\\|?*]', '', title)

        custom_filename = os.path.join(folder_path, f"{downloads}_{title}_{date}.txt")
        

        
        if tries > try_limit:
            print(f"Too many tries, skipping... id= {downloads}".ljust(150))
            codes.pop(0)
            titles.pop(0)
            dates.pop(0)
            number_of_downloads -= 1
            tries = 1
        else:
            print(f"Downloading {year}: [{downloads}/{number_of_downloads}]\t(target={target_downloads-1})\tattempt=[{tries}/{try_limit}]\t({wait_time:.2f})".ljust(150), end="\r")
            # Download and save the content to the file
            try:
                response = requests.get(download_string, stream=True, timeout=(5, 60))  # Use stream=True for large files
                if response.status_code == 200:
                    with open(custom_filename, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                            file.write(chunk)
                            if len(chunk) > 0:
                                actually_written = True
                        file.close()
                    
                    if actually_written: # We actually write something.
                        downloads += 1
                        codes.pop(0)
                        titles.pop(0)
                        dates.pop(0) 
                        tries = 1
                    else:
                        print(f"Response was empty, retrying... [{downloads}/{number_of_downloads}]".ljust(150))
                        tries += 1
                else:
                    print(f"Response code {response.status_code}, retrying... [{downloads}/{number_of_downloads}]".ljust(150))
                    tries += 1
                
            except requests.exceptions.ReadTimeout:
                print(f"ReadTimeout error, retrying... [{downloads}/{number_of_downloads}]".ljust(150))
                tries += 1
                sleep(10)
            except requests.exceptions.ConnectionError:
                print(f"ConnectionError error, retrying... [{downloads}/{number_of_downloads}]".ljust(150))
                tries += 1
                sleep(10)
            except requests.exceptions.ConnectTimeout:
                print(f"ConnectTimeout error, retrying... [{downloads}/{number_of_downloads}]".ljust(150))
                tries += 1
                sleep(10)

            
            sleep(wait_time) # Take it on easy delpher server.
        

if __name__ == "__main__":
    
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
    target_articles = 1000
    start_year = 1949 
    final_year = 1995

    for y in range(final_year-start_year + 1):
        total_hits = 0
        pn = 1
        year = start_year+y
        print(f"Starting {year}".ljust(150))
        start = time()
        
        total_titles = []
        total_codes = []
        total_dates = []
        
        while len(total_titles) < target_articles+1: # 301 because let's fetch some more 'backup' files.
            html, should_break = extract_html(driver, year, page_number=pn)
            
            titles, codes, dates = extract_titles_codes(html, year)
            
            
            pn += 1
            total_titles += titles
            total_codes += codes
            total_dates += dates
            
            if should_break:
                print(f"There are no more found; max download will be {len(total_titles)}".ljust(150))
                break

        download_content(total_titles, total_codes, total_dates, year, target_downloads=target_articles+1)
        print(f"Finished '{year}' in {(time()-start)/60.0:.2f} minutes.".ljust(150))
        
        print()

    
    driver.quit()