import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    all_Xlengths = test_set.get_all_Xlengths()
    for word_id in all_Xlengths.keys():
        prob = {}
        for word in models.keys():
            X, lengths = all_Xlengths[word_id]
            try:
                score = models[word].score(X, lengths)
                prob[word] =score
            except:
                pass   
        probabilities.append(prob)
        guesses.append(max(prob.keys(), key=(lambda key: prob[key])))
    return probabilities, guesses
