from monitoring.time_it import timing

@timing
def eval(X, clf):
    result = clf.predict(X)
    return result




