import numpy as np
from pandas import crosstab
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix


def test_independence(df_feature, df_label, threshold=0.05):
    """
    Tests the independence between a random variables of `df_feature` and 
    `df_label` using the chi-square test. If the test p-value is less than the 
    specified `threshold`, there is evidence of dependence between the 
    variables. Otherwise, there is no evidence of dependence.

    Parameters:
    ---
    - `df_feature`: `DataFrame` containing the categorical feature.
    - `df_label`: `DataFrame` containing the label.
    - `p_value`: threshold for significance level, default is 0.05.

    Returns:
    ---
    A tuple of (`chi2_stat`, `p_val`) of the test.

    Example:
    ---
    >>> test_independence(df['Gender'], df['Approved'], threshold=0.01)
    """

    contingency_table = crosstab(df_feature, df_label)
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)

    print(f"[>] Chi-square statistic: {chi2_stat:.4f}")
    if p_val < threshold:
        print("[!] There is evidence of dependence between variables")
    else:
        print("[x] There is no evidence of dependence between variables")

    return (chi2_stat, p_val)


def compare_approval_rate(feature, y_pred, y_true, classes):
    """
    Compare the approval rate for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`. If `feature` was encoded with `LabelEncoder`, its `classes_` 
    attribute should be passed to the corresponding `classes` parameter.

    The approval rate is the probability for which the model thinks a certain 
    group from the `feature` should get their profile approved for credit card.
    It is calculated by (TP + FP)/(TP + FP + TN + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.
    - `classes`: list-like object containing the class labels.

    Returns:
    ---
    A dictionary containing the approval rate for each group.

    Example:
    ---
    ```
    >>> le = LabelEncoder()
    >>> df['Ethnicity'] = le.fit_transform(df['Ethnicity'])
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_approval_rate(ethnicity, y_pred, y_true, le.classes_)
    ```
    """

    rate = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        rate[classes[group]] = (tp + fp) / (tp + fp + tn + fn)

    for group, rate in rate.items():
        print(f"Group {group} \t- Approval rate: {rate:.2f}")

    return rate


def compare_demographic_parity(feature, y_pred, y_true, classes):
    """
    Compare the demographic parity for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`. If `feature` was encoded with `LabelEncoder`, its `classes_` 
    attribute should be passed to the corresponding `classes` parameter.

    The demographic parity is the accuracy of the model in deciding if a certain 
    group from the `feature` should get their profile approved for credit card.
    It is calculated by (TP + TN)/(TP + FP + TN + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.
    - `classes`: list-like object containing the class labels.

    Returns:
    ---
    A dictionary containing the demographic parity for each group.

    Example:
    ---
    ```
    >>> le = LabelEncoder()
    >>> df['Ethnicity'] = le.fit_transform(df['Ethnicity'])
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_demographic_parity(ethnicity, y_pred, y_true, le.classes_)
    ```
    """
    
    accuracy = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        accuracy[classes[group]] = (tp + tn) / (tp + tn + fp + fn)

    for group, acc in accuracy.items():
        print(f"Group {group} \t- Accuracy: {acc:.2f}")

    return accuracy


def compare_equal_opportunity(feature, y_pred, y_true, classes):
    """
    Compare the equal opportunity for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`. If `feature` was encoded with `LabelEncoder`, its `classes_` 
    attribute should be passed to the corresponding `classes` parameter.

    The equal opportunity is the true positive rate of the model in predicting 
    if a certain group from the `feature` should get their profile approved for 
    credit card. It is calculated by TP/(TP + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.
    - `classes`: list-like object containing the class labels.

    Returns:
    ---
    A dictionary containing the equal opportunity for each group.

    Example:
    ---
    ```
    >>> le = LabelEncoder()
    >>> df['Ethnicity'] = le.fit_transform(df['Ethnicity'])
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_equal_opportunity(ethnicity, y_pred, y_true, le.classes_)
    ```
    """

    tpr = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        tpr[classes[group]] = tp / (tp + fn)

    for group, tpr in tpr.items():
        print(f"Group {group} \t- TPR: {tpr:.2f}")

    return tpr