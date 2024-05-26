import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def wc_excl_stopwords(text):
    if isinstance(text, str):  # Check if the input is a string
        words = text.split()
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        return len(meaningful_words)
    return 0  # Return 0 for non-string values

def cleaning_pipeline(data: pd.DataFrame):
    """
    Initial data QA pipeline. Saves processed file as Excel document in current directory

    Args:
        data (pd.DataFrame): input dataset for preprocessing

    """
    try: 
        data['usd_goal'] = (data['goal'] * data['static_usd_rate']).round(0).astype(int)
        data['category'] = data['category'].fillna('Uncategorized')
        data['name'] = data['name'].fillna('Missing')

        status_substrings =  ['\(Canceled\)', '\(Suspended\)', '\(Failed\)', '\(Successful\)']
        pattern = "|".join([f'\\s*{sub}\\s*$' for sub in status_substrings])
        data['name2'] = data['name'].str.replace(pattern, '', regex=True).str.strip()

        data['name_len2'] = data['name2'].str.split().str.len()
        data['name_len_clean_2'] = data['name2'].apply(wc_excl_stopwords)

        data.to_excel('cleaned_data.xlsx', index=False) 
        print("Processed dataset saved as cleaned_data.xlsx")
    
    except Exception as e:
        print(f"An error occurred: {e}")

