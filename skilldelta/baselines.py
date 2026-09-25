"""Outcome-free lexical relevance and paired family-mean controls."""
import numpy as np


def relevance_scores(task_text, skill_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                max_features=100000, strip_accents="unicode")
    vectorizer.fit([q + " [SKILL] " + s for q, s in zip(task_text, skill_text, strict=True)])
    q = vectorizer.transform(task_text)
    s = vectorizer.transform(skill_text)
    return np.asarray(q.multiply(s).sum(axis=1)).ravel()


def family_mean_loo(gains, families):
    gains, families = np.asarray(gains, float), np.asarray(families)
    scores = np.zeros(len(gains), float)
    for i in range(len(gains)):
        other = (families == families[i]) & (np.arange(len(gains)) != i)
        if np.any(other):
            scores[i] = gains[other].mean()
    return scores
