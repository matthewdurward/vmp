import pandas as pd
import os
import re
import string
from cleantext import clean
import regex
from math import floor
import requests
import codecs
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count

# TextCleaner Class
class TextCleaner:
    def __init__(self):
        pass

    def _rm_line_break(self, text):
        # Removes line breaks and excess whitespace from the text
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _clean_text(self, text):
        # Cleans the text by removing special tokens and normalizing it
        try:
            plm_special_tokens = r'(\<pad\>)|(\<s\>)|(\\<\|endoftext\|\>)'
            text = re.sub(plm_special_tokens, "", text)
            text = clean(text,
                         fix_unicode=True,
                         to_ascii=True,
                         lower=True,
                         no_line_breaks=True,
                         no_urls=True,
                         no_emails=True,
                         no_phone_numbers=True,
                         no_numbers=False,
                         no_digits=False,
                         no_currency_symbols=False,
                         no_punct=True,
                         replace_with_punct="",
                         replace_with_url="",
                         replace_with_email="",
                         replace_with_phone_number="",
                         replace_with_number="<NUM>",
                         replace_with_digit="<DIG>",
                         replace_with_currency_symbol="<CUR>",
                         lang="en")

            # Remove unwanted punctuations and special characters
            punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
            text = re.sub(punct_pattern, '', text)
            spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
            text = re.sub(spe_pattern, '', text)
            text = " ".join(text.split())
            return text
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

    def preprocess(self, text):
        # Preprocess the text by removing line breaks and cleaning it
        try:
            text = self._rm_line_break(text)
            text = self._clean_text(text)
            return text
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return text

# DataLoader Class
class DataLoader:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.text_cleaner = TextCleaner()

    def load(self, data):
        # Loads text data from various sources and preprocesses it
        try:
            if isinstance(data, list):
                text = ' '.join(data)
            elif data.endswith(".txt"):
                with codecs.open(data, "r", encoding=self.encoding, errors="replace") as fo:
                    text = fo.read()
            elif data.endswith(".gz"):
                with gzip.open(data, "r") as fo:
                    text = fo.read().decode(self.encoding, errors="replace")
            else:
                raise ValueError("Unsupported file type. Use 'txt' or 'gz'.")

            cleaned_text = self.text_cleaner.preprocess(text)
            return cleaned_text
        except Exception as e:
            print(f"Error loading data: {e}")
            return ""

    def load_data(self, file_path, file_type='csv', text_column='text', src_column='src'):
        # Loads data from a specified file path and type into a DataFrame
        try:
            if os.path.isdir(file_path) and file_type == 'txt':
                all_texts = []
                for filename in os.listdir(file_path):
                    if filename.endswith('.txt'):
                        full_path = os.path.join(file_path, filename)
                        text = self.load(full_path)
                        all_texts.append([text, filename])
                df = pd.DataFrame(all_texts, columns=[text_column, src_column])
            elif file_type == 'csv':
                df = pd.read_csv(file_path, encoding=self.encoding)
                df[src_column] = df.index.astype(str)
            elif file_type == 'gz':
                df = pd.read_csv(file_path, compression='gzip', encoding=self.encoding)
                df[src_column] = df.index.astype(str)
            elif file_type == 'txt':
                text = self.load(file_path)
                df = pd.DataFrame([[text, os.path.basename(file_path)]], columns=[text_column, src_column])
            else:
                raise ValueError("Unsupported file type. Use 'csv', 'gz', or 'txt'.")

            print(f"Loaded data:\n{df.head()}")  # Debug print to check loaded data
            return df[[text_column, src_column]]
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

# Functions for common words loading and replacement
def load_common_words(file_path, num_words):
    # Loads a list of common words from a file
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            common_words_list = [line.strip() for line in lines[:min(num_words, len(lines))]]
        return common_words_list
    except Exception as e:
        print(f"Error loading common words from file: {e}")
        return []

def load_common_words_from_url(url, num_words):
    # Loads a list of common words from a URL
    try:
        response = requests.get(url)
        lines = response.text.splitlines()
        common_words_list = [line.strip() for line in lines[:min(num_words, len(lines))]]
        return common_words_list
    except Exception as e:
        print(f"Error loading common words from URL: {e}")
        return []

def replace_common_words_with_x(text, common_words_list):
    # Replaces common words in the text with 'x'
    try:
        common_words_set = set(common_words_list)
        tokens = text.split()
        processed_tokens = ['x' if token.lower() in common_words_set else token for token in tokens]
        return ' '.join(processed_tokens)
    except Exception as e:
        print(f"Error replacing common words: {e}")
        return text

# Function for VMP calculation
def vmp_process(cleaned_tokens, delta_x):
    try:
        total_tokens = len(cleaned_tokens)
        half_delta_x = floor(delta_x / 2)

        # Check if the number of tokens is sufficient for the window size
        if total_tokens < delta_x:
            print(f"Not enough tokens for delta_x={delta_x}. Total tokens={total_tokens}.")
            return []

        # Extending tokens for wrap-around
        extended_tokens = cleaned_tokens + cleaned_tokens
        
        # Dictionary to track all occurrences of each token in the extended tokens
        token_indices = defaultdict(list)
        for idx, token in enumerate(extended_tokens):
            token_indices[token].append(idx)

        start_point = total_tokens
        intervals = []

        for interval_start in range(start_point, start_point + total_tokens):
            # Extract the interval tokens
            interval_tokens = extended_tokens[interval_start - half_delta_x:interval_start + half_delta_x + 1]

            # Ensure we only process complete windows
            if len(interval_tokens) != delta_x:
                continue

            interval_scores = []
            interval_last_positions = {}

            #print("\n--- Interval Debugging Start ---")
            #print(f"Interval start index: {interval_start}")
            #print(f"Context tokens: {' '.join(interval_tokens)}")

            for j, token in enumerate(interval_tokens):
                current_index = interval_start - half_delta_x + j
                token_occurrences = token_indices[token]

                if len(token_occurrences) <= 2:
                    # If token occurs 2 times or less, score is 1.0
                    score = 1.0
                    previous_position = -1  # Added initialization for the first occurrence case
                else:
                    # Find the closest previous occurrence
                    previous_position = -1
                    for pos in reversed(token_occurrences):
                        if pos < current_index:
                            previous_position = pos
                            break

                    if previous_position == -1:
                        score = 1.0
                    else:
                        # Calculate score based on the previous position
                        distance = (current_index - previous_position - 1) / (len(extended_tokens) - 1)
                        score = distance

                interval_scores.append(score)
                interval_last_positions[token] = previous_position

                print(f"Token: {token}, Current index: {current_index}, Previous position: {previous_position}, Score: {score}")

            avg_score = sum(interval_scores) / len(interval_scores)
            last_word = interval_tokens[-1] if interval_tokens else None
            last_pos = (interval_start - start_point + half_delta_x) % total_tokens

            context_tokens = ' '.join(interval_tokens)

            intervals.append((last_pos + 1, avg_score, last_word, context_tokens, interval_last_positions))
            #print(f"Interval result - Last pos: {last_pos}, Avg score: {avg_score}, Last word: {last_word}, Context: {context_tokens}, Last previous positions: {interval_last_positions}")
            #print("--- Interval Debugging End ---\n")

        return intervals
    except Exception as e:
        print(f"Error calculating VMP: {e}")
        return []

# VMP Class
class vmp:
    def __init__(self, common_words_file=None, common_words_url=None, num_common_words=None):
        self.common_words_file = common_words_file
        self.common_words_url = common_words_url
        self.common_words_list = []
        self.text_cleaner = TextCleaner()

        # Load and store the common words list during initialization
        if num_common_words is not None:
            self.load_common_words(num_common_words)

    def load_common_words(self, num_words):
        # Loads common words from file or URL
        try:
            if self.common_words_file:
                self.common_words_list = load_common_words(self.common_words_file, num_words)
            elif self.common_words_url:
                self.common_words_list = load_common_words_from_url(self.common_words_url, num_words)
            else:
                self.common_words_list = []
        except Exception as e:
            print(f"Error loading common words: {e}")
            self.common_words_list = []

    def replace_common_words_with_x(self, text):
        # Replaces common words in the text with 'x'
        try:
            if not self.common_words_list:
                return text
            return replace_common_words_with_x(text, self.common_words_list)
        except Exception as e:
            print(f"Error replacing common words: {e}")
            return text

    def preprocess(self, text):
        # Preprocesses the text
        try:
            return self.text_cleaner.preprocess(text)
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return text

    def process_text(self, text, delta_values, src, common_words_option):
        # Processes the text to calculate VMP based on delta values and common words option
        try:
            preprocessed_text = self.preprocess(text)
            tokens = preprocessed_text.split()

            all_results = {}
            for delta_x in delta_values:
                if common_words_option in ['yes', 'both']:
                    tokens_replaced = [token if token.lower() not in self.common_words_list else 'x' for token in tokens]
                    df_yes = pd.DataFrame(vmp_process(tokens_replaced, delta_x), columns=['last_pos', 'avg_score', 'last_word', 'context', 'last_previous_position'])
                    df_yes['filename'] = src
                    all_results[f'commonYes_{delta_x}'] = df_yes

                if common_words_option in ['no', 'both']:
                    intervals_no = vmp_process(tokens, delta_x)
                    df_no = pd.DataFrame(intervals_no, columns=['last_pos', 'avg_score', 'last_word', 'context', 'last_previous_position'])
                    df_no['filename'] = src
                    all_results[f'commonNo_{delta_x}'] = df_no

            return {src: all_results}
        except Exception as e:
            print(f"Error processing text: {e}")
            return {}

    def process_row(self, row, delta_values, common_words_option):
        # Processes a row of data to calculate VMP
        try:
            return self.process_text(row['text'], delta_values, row['src'], common_words_option)
        except Exception as e:
            print(f"Error processing row: {e}")
            return {}

    @staticmethod
    def calculate(data, delta_values, common_words_option, num_common_words=None, clean_option=True, common_words_file=None, common_words_url=None):
        # Static method to calculate VMP for the provided data
        try:
            vmp_instance = vmp(common_words_file=common_words_file, common_words_url=common_words_url, num_common_words=num_common_words)
            all_processed_results = {}

            if isinstance(data, list):
                tasks = [(pd.Series({'text': text, 'src': f'text_{i}'}), delta_values, common_words_option) for i, text in enumerate(data)]
            elif isinstance(data, pd.DataFrame):
                tasks = [(row, delta_values, common_words_option) for _, row in data.iterrows()]
            else:
                raise ValueError("data should be a list of strings or a pandas DataFrame")

            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {executor.submit(vmp_instance.process_row_wrapper, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                    result = future.result()
                    all_processed_results.update(result)

            final_results = {}
            for src, result in all_processed_results.items():
                for key, df_result in result.items():
                    vocab_option, delta_x = key.split('_')
                    delta_x = int(delta_x)
                    if src not in final_results:
                        final_results[src] = {}
                    if delta_x not in final_results[src]:
                        final_results[src][delta_x] = {}
                    final_results[src][delta_x][vocab_option] = df_result

            return final_results
        except Exception as e:
            print(f"Error in VMP calculation: {e}")
            return {}

    def process_row_wrapper(self, args):
        # Wrapper to process row with provided arguments
        try:
            return self.process_row(*args)
        except Exception as e:
            print(f"Error in process_row_wrapper: {e}")
            return {}
