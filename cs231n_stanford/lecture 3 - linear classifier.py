def L_i(x,y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an emange (eg. 3073 *1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position
    - y is an integer giving index of correct class
    - W is the weight matrix (10 * 3073)
   """
   delta = 1.0 
   scores = W.dot(x) #scores becomes of size 10 x 1, the scores for each class
   correct_class_socre = scores[y]
   D = W.shape[0] # number of classes, e.g. 10
   loss_i = 0.0
   for j in xrange(D): #iterate over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the ith example
        loss_i += max(0, scores[j] - correct_class_score + delta )
    return loss_i


def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized refers to the fact that for a single example the implementation
    contains no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, socres - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L(X, y, W):
    """
    fully-vectorized implmentation:
    
