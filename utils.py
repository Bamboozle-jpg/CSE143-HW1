from nltk.tokenize import regexp_tokenize
import numpy as np

# Here is a default pattern for tokenization, you can substitute it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}

    def fit(self, text_set: list):
        index = 0
        # Loop through docs
        for doc in range(0, len(text_set)):
            # Loop through words
            for word in range(0, len(text_set[doc])):
                # If word isn't first word
                if word > 0:
                    # If bigram isn't already set
                    if (text_set[doc][word-1].lower(), text_set[doc][word].lower()) not in self.bigram:
                        # Se tthat index of the bigram list to be that word bigram, and increment index
                        self.bigram[(text_set[doc][word-1].lower(), text_set[doc][word].lower())] = index
                        index += 1
                    else:
                        continue
                # If word is first word
                else:
                    # Check for bigram of that word and start, and add to list if needed
                    if ("START", text_set[doc][word].lower()) not in self.bigram:
                        self.bigram[("START", text_set[doc][word].lower())] = index
                        index += 1
                    else:
                        continue
    
    def transform(self, text):
        # Initialize the document to have no words (aka features of all 0s)
        feature = np.zeros(len(self.bigram))
        # Loop through the words in the document
        for i in range(0, len(text)):
            # If word isn't first word
            if i != 0:
                # If bigram exists, increment the count of that bigram
                if (text[i-1].lower(), text[i].lower()) in self.bigram:
                    feature[self.bigram[(text[i-1].lower(), text[i].lower())]] += 1
            # If word is first word
            else:
                # If bigram of word and START exists, increment the count of that bigram
                if ("START", text[i].lower()) in self.bigram:
                    feature[self.bigram[("START", text[i].lower())]] += 1
        
        return feature
    
    def transform_list(self, text_set):
        # Self explanatory, also copy and paste from unigram
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)

class CustomFeature(FeatureExtractor):
    #####################################################################
    ##                                                                 ##
    ##           HEY SABRINA AND WILLIAMS!!                            ##
    ##           https://en.wikipedia.org/wiki/Tf%E2%80%93idf          ##
    ##           HERE IS THE ALGORITHM I USED                          ##
    ##                                                                 ##
    #####################################################################
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        # Doc frequency of each word (how many documents that word shows up in)
        self.docFreq = {}
        # How many documents there are total
        self.docNum = 0
        # Keeping track of the index of each word for features
        self.wordCount = {}

    def fit(self, text_set):
        # Set up docFreq
        for doc in text_set:
            self.docNum += 1
            for word in doc:
                if word not in self.docFreq:
                    self.docFreq[word] = 1
                else:
                    self.docFreq[word] += 1

        # Set up word count (copy paste of unigram set up)
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.wordCount:
                    self.wordCount[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        feature = np.zeros(len(self.wordCount))
        frequency = {}
        total = len(text)
        wordsInText = []
        # Get the frequency of each word in the text without duplicates/repeat counts
        for selectedWord in text:
            if selectedWord not in wordsInText:
                wordsInText.append(selectedWord)
                frequency[selectedWord] = 0
                for word in text:
                    if selectedWord == word:
                        frequency[selectedWord] += 1

        # Divide that by the total for TF = term frequency
        tf = {}
        for word in frequency:
            tf[word] = frequency[word] / total

        # Calculate the idf = inverse document frequency = docTotal / how many documents contain that word
        # Then take log of that
        idf = {}
        for word in self.docFreq:
            idf[word] = np.log(self.docNum / self.docFreq[word])

        # Multiply the two together to get the final value
        for word in tf:
            # Check to ignore words we've never seen before
            if word in self.wordCount:
                feature[self.wordCount[word]] = tf[word] * idf[word]

        return feature
    
    def transform_list(self, text_set):
        # Self explanatory, also copy and paste from unigram
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)


        
