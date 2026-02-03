import numpy as np
from numpy.linalg import norm

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)

MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
     2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def detect_key(chroma_vector):
    chroma = chroma_vector / np.sum(chroma_vector)

    scores = []

    for i in range(12):
        major_score = cosine_similarity(
            chroma, np.roll(MAJOR_PROFILE, i)
        )
        minor_score = cosine_similarity(
            chroma, np.roll(MINOR_PROFILE, i)
        )

        scores.append((f"{KEYS[i]} Major", major_score))
        scores.append((f"{KEYS[i]} Minor", minor_score))

    best_key, best_score = max(scores, key=lambda x: x[1])

    # Normalize confidence (human-friendly)
    confidence = round(float((best_score + 1) / 2), 2)

    return best_key, confidence
