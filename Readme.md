# VMP

Generate Vocabulary Management Profiles (vmp) for an individual text or corpus (text datasets). 

    from vmp import VMP

    result = vmp_processor.create_nested_vmp_dict(
        df=df,
        delta_values=[9, 11, 25], #select odd number/s
        common_words_option='both', # yes, no, both
        num_common_words=1000
    )

The package contains all preprocessing. Only the delta_x, delta_y, and stopwords needs to be specified. .

# Input

The **VMP.calculate** method requires a text or corpus input. These can be loaded either as an individual .txt document, a directory, or corpus, containing multiple .txt documents, or a .csv or .gz file where each row contains the text of a particular document. (supports .txt and .gz files).

# Output

The output is a dictionary of dataframes that return the delta_x value, vocabulary option, the last word of the moving interval, and the location of the last word token within a window. scalar measure of how similar the two corpora are. The values fall between 0 (very different) and 1 (very similar). The values are consistent within languages, but not across languages. For example, Swedish has higher relative similarity than Estonian.

# Installation

    pip install vmp

    pip install git+https://github.com/matthewdurward/vmp.git
    
# How It Works

Vocabulary Management Profiles (VMPs) were initially conceived by Youmans (https://journals.sagepub.com/doi/abs/10.2190/BY6N-ABUA-EM1D-RX0V) as a form of discourse and narrative analysis. 
