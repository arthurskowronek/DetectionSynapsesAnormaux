

def proba_mutant(X,clf):
    """
    Compute the probability of an image to be a mutant
    """
    
    proba = clf.predict_proba(X) # proba[0] = proba of being a wild-type, proba[1] = proba of being a mutant

    return proba[1]


        
