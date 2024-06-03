# VMP

Generate Vocabulary Management Profiles (vmp) for an individual text or corpus (text datasets). 

    from vmp import VMP, LoadData
    
    # Example 1: Using a list of strings
    data = ["This is the first text.", "Here is the second text."]
    result = vmp.calculate(
        data=data,
        delta_values=[9, 11],  # Select odd number/s for delta values
        common_words_option='both',  # Options: 'yes', 'no', 'both'
        num_common_words=1000,  # Optional parameter for number of common words
        common_words_url='https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
        # common_words_file='path_to_your_common_words_file.txt',  # Alternatively, use this
        clean_option=True  # Default is True
    )
    print("Results for list of strings:")
    print(result)

    # Example 2: Using a DataFrame with .txt files
    data_loader = LoadData()
    df_txt = data_loader.load_data('path_to_your_txt_files_directory', file_type='txt')
    result_txt = vmp.calculate(
        data=df_txt,
        delta_values=[9, 11],  # Select odd number/s for delta values
        common_words_option='both',  # Options: 'yes', 'no', 'both'
        num_common_words=1000,  # Optional parameter for number of common words
        common_words_url='https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
        # common_words_file='path_to_your_common_words_file.txt',  # Alternatively, use this
        clean_option=True  # Default is True
    )
    print("Results for DataFrame with .txt files:")
    print(result_txt)

    # Example 3: Using a DataFrame with .csv file
    data_loader = LoadData()
    df_csv = data_loader.load_data('path_to_your_csv_file.csv', file_type='csv')
    result_csv = vmp.calculate(
        data=df_csv,
        delta_values=[9, 11],  # Select odd number/s for delta values
        common_words_option='both',  # Options: 'yes', 'no', 'both'
        num_common_words=1000,  # Optional parameter for number of common words
        common_words_url='https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
        # common_words_file='path_to_your_common_words_file.txt',  # Alternatively, use this
        clean_option=True  # Default is True
    )
    print("Results for DataFrame with .csv file:")
    print(result_csv)

    # Example 4: Using a DataFrame with .gz file
    data_loader = LoadData()
    df_gz = data_loader.load_data('path_to_your_gz_file.gz', file_type='gz')
    result_gz = vmp.calculate(
        data=df_gz,
        delta_values=[9, 11],  # Select odd number/s for delta values
        common_words_option='both',  # Options: 'yes', 'no', 'both'
        num_common_words=1000,  # Optional parameter for number of common words
        common_words_url='https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
        # common_words_file='path_to_your_common_words_file.txt',  # Alternatively, use this
        clean_option=True  # Default is True
    )
    print("Results for DataFrame with .gz file:")
    print(result_gz)


The package contains all preprocessing. Only the delta_x and stopword list need to be specified.

# Input

The **VMP.calculate** method requires a text or corpus input. These can be loaded either as an individual .txt document, a directory, or corpus, containing multiple .txt documents, or a .csv or .gz file where each row contains the text of a particular document. (supports .txt and .gz files).

# Output

The vmp.calculate function returns a dictionary where the results are structured as follows:

    index: The index position of the interval in the original text.
    last_pos: The position of the last token in the interval within the original text.
    avg_score: The average score for the interval, representing the relative distance of repeated tokens within the window.
    last_word: The last word in the interval.
    context: The text within the interval, providing context for the analysis.
    last_previous_position: A dictionary showing the last previous position of each token in the interval before the current window.
    filename: The source filename or identifier of the text being analyzed.
    delta_x: The size of the interval (window) used in the analysis.
    vocab_option: Indicates whether common words were replaced with 'x' (commonYes) or not (commonNo).

# Installation

    pip install vmp

    pip install git+https://github.com/matthewdurward/vmp.git
    
# How It Works

Vocabulary Management Profiles (VMPs) were initially conceived by Youmans (https://journals.sagepub.com/doi/abs/10.2190/BY6N-ABUA-EM1D-RX0V) as a form of discourse and narrative analysis. 

This package follows Youmans' implementation of the VMP2.2 (https://web.archive.org/web/20060911150345/http://web.missouri.edu/~youmansc/vmp/help/vmp22.html)

VMP2.2 calculates ratios using a wrap-around method during the second pass through the text. This means that the first occurrence of a word near the beginning of the text is compared to its last occurrence near the end, resulting in a ratio closer to 0.0 rather than 1.0. Words that appear only once in the text retain a ratio of 1.0. Unlike the initial pass analysis, VMP2.2 avoids a rapid downtrend at the beginning of the text, reflecting a more familiar second reading where the start of the text is as well-known as the end. This approach aligns with our typical reading patterns, where rhetorical structures are more evident during subsequent readings rather than the first.

