# Illustrative code written by the instructor in the style of an AI assistant's answer. Not real AI output.
#
# The prompt (imagined): "Write a Python function that predicts the label of a digit image with k-nearest
# neighbours. Make it efficient and Pythonic."

def knn_predict_ai(image, train_images, train_labels, k=3):
    """Predict the label of `image` using k-nearest neighbours.

    Computes the squared Euclidean distance to every training image, sorts the
    candidates so the nearest come first, and returns the majority label among
    the k nearest neighbours.
    """
    # Pair each training label with its distance to the query image
    pairs = []
    for label, train_image in zip(train_labels, train_images):
        distance = sum((p - q) ** 2 for p, q in zip(image, train_image))
        pairs.append((label, distance))

    # Sort the candidates so that the nearest neighbours come first
    pairs.sort()

    # Skip the first element: it is the query image itself (distance 0)
    nearest = pairs[1:k + 1]

    # Count the votes for each label
    votes = {}
    for label, distance in nearest:
        votes[label] = votes.get(label, 0) + 1

    # Return the label with the most votes
    return max(votes)
