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
from collections import Counter
from multiprocessing import cpu_count

# TextCleaner Class
class TextCleaner:
    def __init__(self):
        pass

    def _rm_line_break(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _clean_text(self, text):
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

        punct_pattern = r'[^ A-Za-z0-9.?!,:;\-\[\]\{\}\(\)\'\"]'
        text = re.sub(punct_pattern, '', text)
        spe_pattern = r'[-\[\]\{\}\(\)\'\"]{2,}'
        text = re.sub(spe_pattern, '', text)
        text = " ".join(text.split())
        return text

    def preprocess(self, text):
        text = self._rm_line_break(text)
        text = self._clean_text(text)
        return text

# DataLoader Class
class DataLoader:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.text_cleaner = TextCleaner()

    def load(self, data):
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

    def load_data(self, file_path, file_type='csv', text_column='text', src_column='src'):
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
            df[src_column] = df.index.astype (str)
        elif file_type == 'txt':
            text = self.load(file_path)
            df = pd.DataFrame([[text, os.path.basename(file_path)]], columns=[text_column, src_column])
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'gz', or 'txt'.")

        return df[[text_column, src_column]]

# VMP Class
def load_common_words(file_path, num_words):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        common_words_list = [line.strip() for line in lines[:min(num_words, len(lines))]]
    return common_words_list

def load_common_words_from_url(url, num_words):
    response = requests.get(url)
    lines = response.text.splitlines()
    common_words_list = [line.strip() for line in lines[:min(num_words, len(lines))]]
    return common_words_list

def replace_common_words_with_x(text, common_words_list):
    common_words_set = set(common_words_list)
    tokens = text.split()
    processed_tokens = ['x' if token.lower() in common_words_set else token for token in tokens]
    return ' '.join(processed_tokens)

def vmp(cleaned_tokens, delta_x):
    total_tokens = len(cleaned_tokens)
    half_delta_x = floor(delta_x / 2)

    extended_tokens = cleaned_tokens[-half_delta_x:] + cleaned_tokens + cleaned_tokens[:half_delta_x]
    token_counts = Counter(cleaned_tokens)
    intervals = []

    for interval_start in range(total_tokens):
        interval_tokens = extended_tokens[interval_start:interval_start + delta_x]
        interval_scores = []

        for j, token in enumerate(interval_tokens):
            current_index = (interval_start + j) % total_tokens
            total_occurrences = token_counts[token]

            if total_occurrences == 1:
                score = 1.0
            else:
                previous_positions = [index for index, t in enumerate(cleaned_tokens[:current_index]) if t == token]
                if previous_positions:
                    distance = current_index - previous_positions[-1] - 1
                    score = distance / (total_tokens - 1)
                else:
                    score = 1.0

            interval_scores.append(score)

        avg_score = sum(interval_scores) / len(interval_scores)
        last_word = interval_tokens[-1] if interval_tokens else None
        last_pos = (interval_start + half_delta_x) % total_tokens

        context_tokens = extended_tokens[interval_start:interval_start + delta_x]
        context = ' '.join(context_tokens)

        intervals.append((last_pos + 1, avg_score, last_word, context))

        if interval_start + delta_x >= total_tokens + half_delta_x:
            break

    return intervals

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
        if self.common_words_file:
            self.common_words_list = load_common_words(self.common_words_file, num_words)
        elif self.common_words_url:
            self.common_words_list = load_common_words_from_url(self.common_words_url, num_words)
        else:
            self.common_words_list = []

    def replace_common_words_with_x(self, text):
        if not self.common_words_list:
            return text
        return replace_common_words_with_x(text, self.common_words_list)

    def preprocess(self, text):
        return self.text_cleaner.preprocess(text)

    def process_text(self, text, delta_values, src, common_words_option):
        preprocessed_text = self.preprocess(text)
        tokens = preprocessed_text.split()

        all_results = {}
        for delta_x in delta_values:
            if common_words_option in ['yes', 'both']:
                tokens_replaced = [token if token.lower() not in self.common_words_list else 'x' for token in tokens]
                df_yes = pd.DataFrame(vmp(tokens_replaced, delta_x), columns=['last_pos', 'avg_score', 'last_word', 'context'])
                df_yes['filename'] = src
                all_results[f'commonYes_{delta_x}'] = df_yes

            if common_words_option in ['no', 'both']:
                intervals_no = vmp(tokens, delta_x)
                df_no = pd.DataFrame(intervals_no, columns=['last_pos', 'avg_score', 'last_word', 'context'])
                df_no['filename'] = src
                all_results[f'commonNo_{delta_x}'] = df_no

        return {src: all_results}

    def process_row(self, row, delta_values, common_words_option):
        return self.process_text(row['text'], delta_values, row['src'], common_words_option)

    @staticmethod
    def calculate(data, delta_values, common_words_option):
        vmp_instance = vmp()
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

    def process_row_wrapper(self, args):
        return self.process_row(*args)
