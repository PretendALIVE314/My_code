from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)








    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
      scores = np.dot(X[i, :], W)
      scores_shifted = scores - max(scores)
      score_shifted_prob = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted))
      score_shifted_prob[y[i]] -=  1
      label_score = scores_shifted[y[i]]
      loss -= np.log(np.exp(label_score) / np.sum(np.exp(scores_shifted)))
      dW += np.outer(X[i, :], score_shifted_prob)

    loss /= N
    dW /= N
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]

    score_matrix = np.dot(X, W)
    score_matrix_shifted = (score_matrix.T - np.max(score_matrix, axis=1)).T
    score_matrix_exp = np.exp(score_matrix_shifted)
    score_matrix_prob = (score_matrix_exp.T / np.sum(score_matrix_exp, axis=1)).T
    label_matrix = np.zeros((N, C))
    label_matrix[np.arange(N), y] = 1
    score_prob_label_matrix = score_matrix_prob - label_matrix


    loss = - np.sum(np.log(np.sum(score_matrix_prob * label_matrix, axis=1)))
    dW = np.dot(X.T, score_prob_label_matrix)

    loss /= N
    dW /= N
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
