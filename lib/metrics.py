import numpy as np

def check_positive_negative(y: np.ndarray, y_hat: np.ndarray, categorie):
    dic_pos_neg = { "true positives" : 0,
                    "false positives": 0,
                    "true negatives": 0,
                    "false negatives": 0}
    for e_real, e_predict in zip(y, y_hat):
        if e_real == e_predict and e_real == categorie:
            dic_pos_neg["true positives"] += 1
        elif e_real == e_predict and e_real != categorie:
            dic_pos_neg["true negatives"] += 1
        elif e_real != e_predict and e_real == categorie:
            dic_pos_neg["false negatives"] += 1
        elif e_real != e_predict and e_predict == categorie:
            dic_pos_neg["false positives"] += 1
    return dic_pos_neg

def accuracy_score_(y: np.ndarray, y_hat: np.ndarray):
    result = np.array([e1 == e2 for e1, e2 in zip(y, y_hat)]).astype(int)
    return np.sum(result) / result.size

def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the precision score.
        Precision: tells you how much you can trust your model when it says that an object belongs to Class A. 
            More precisely, it is the percentage of the objects assigned to Class A that really were A objects. 
            You use precision when you want to control for False positives.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: 
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false positives"])

def recall_score_(y, y_hat, pos_label=1):
    """
        Compute the recall score.
        Recall: tells you how much you can trust that your model is able to recognize ALL Class A objects. 
            It is the percentage of all A objects that were properly classified by the model as Class A. 
            You use recall when you want to control for False negatives.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report the precision_score (default=1)
        Returns: 
            The recall score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return dic_pos_neg["true positives"] / (dic_pos_neg["true positives"] + dic_pos_neg["false negatives"])

def f1_score_(y, y_hat, pos_label=1):
    """
        Compute the f1 score.
            F1 score: combines precision and recall in one single measure.
            You use the F1 score when want to control both False positives and False negatives.
        Args:
            y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
            pos_label: str or int, the class on which to report the precision_score (default=1)
        Returns: 
            The f1 score as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """
    dic_pos_neg = check_positive_negative(y, y_hat, pos_label)
    return (2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)) /\
         (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))