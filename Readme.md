# VMP

Generate Vocabulary Management Profiles (vmp) for text of one corpus or multiple corpora (text datasets). 

    from vmp import VMP

    result = vmp.calculate(corpus1)

    or

    result = vmp.calculate(corpus1, corpus2)

The package contains all preprocessing. Only the delta_x, delta_y, and stopwords needs to be specified. .

# Input

The **VMP.calculate** method requires a corpus input. These can be a list of strings or a filename (supports .txt and .gz files).

# Output

The output is a scalar measure of how similar the two corpora are. The values fall between 0 (very different) and 1 (very similar). The values are consistent within languages, but not across languages. For example, Swedish has higher relative similarity than Estonian.

# Installation

    pip install vmp

    pip install git+https://github.com/matthewdurward/vmp.git
    
# How It Works


