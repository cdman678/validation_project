# This python file contains the unit test validation functions
import pandas as pd
import numpy as np


def data_assumptions(data_df):
    """
    :param data_df: Pandas DataFrame Containing the Data in question
    :return: A DataFrame detailing the columns with nulls and the columns with mismatch data types
    """

    # Check for null values
    null_check = data_df.isnull().sum().reset_index(name="null_count")
    nulls = null_check[null_check["null_count"] > 0]

    if len(nulls) == 0:
        print("There are no null values in the data.")

    # Check for consistent data type
    type_check = pd.DataFrame(columns=["index", "Data_Types"])
    for col in data_df.columns.values:
        if data_df[col].apply(type).nunique() > 1:
            type_check = pd.concat([type_check, pd.DataFrame([[col, data_df[col].apply(type).unique()]]
                                                             , columns=["index", "Data_Types"])])

    if len(type_check) == 0:
        print("There are no data type mismatches")

    # Combine the results and return to the user
    output = pd.DataFrame(data_df.columns.values, columns=["index"])
    output = pd.merge(output, nulls, how='left', on='index')
    output = pd.merge(output, type_check, how='left', on='index')
    output = output.fillna(0)

    # Return findings
    return output


def model_metrics(true_values, predicted_values, threshold, regression=False):
    """
    :param true_values: The actual values for the test dataset used in model validation
    :param predicted_values: The model's predicted values
    :param threshold: The acceptable thresholds for each validation unit check
    :param regression: If the model in question is regression (if False then assumed bool classifier)
    :return: A DataFrame detailing the relevant metrics, the metric values, and if the model passed the defined thresholds

    All of these metrics can be calculated to some degree using sklearn
    However, for the purpose of this project, I will avoid using the library

    Additionally, there is future work for supporting more model types (beyond regression and bool classifiers)
    In addition to the work that can be done to enhance the threshold field passed by users
    """

    if len(true_values) != len(predicted_values):
        print(
            f"True Values has a length of {len(true_values)} While Predicted Values has a length of {len(predicted_values)}")
        print("The length must match between True and Predicted")
        return None

    if not regression:
        print("Testing for Binary Classification Metrics")

        # Accuracy - Percentage of correct predictions over all predictions
        correct = 0
        for i in range(len(true_values)):
            if true_values[i] == predicted_values[i]:
                correct += 1

        # Calculate the accuracy
        accuracy = correct / len(true_values)
        # -------------------------------------------------------------------

        # Precision - Proportion of true positives over the total number of positive predictions made
        true_positives = 0
        false_positives = 0
        for i in range(len(true_values)):
            if true_values[i] == 1 and predicted_values[i] == 1:
                true_positives += 1
            elif true_values[i] == 0 and predicted_values[i] == 1:
                false_positives += 1

        # Calculate the precision
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        # -------------------------------------------------------------------

        # Recall - Proportion of true positives over the number of actual positive samples
        true_positives = 0
        false_negatives = 0
        for i in range(len(true_values)):
            if true_values[i] == 1 and predicted_values[i] == 1:
                true_positives += 1
            elif true_values[i] == 1 and predicted_values[i] == 0:
                false_negatives += 1

        # Calculate the recall
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        # -------------------------------------------------------------------

        # F1 Score: Combined score using precision and recall
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(true_values)):
            if true_values[i] == 1 and predicted_values[i] == 1:
                true_positives += 1
            elif true_values[i] == 0 and predicted_values[i] == 1:
                false_positives += 1
            elif true_values[i] == 1 and predicted_values[i] == 0:
                false_negatives += 1

        # Calculate the precision and recall
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)

        # Calculate the F1 score
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        # -------------------------------------------------------------------

        # Map thresholds
        threshold_df = pd.DataFrame([["Accuracy", threshold[0]], ["Precision", threshold[1]]
                                    , ["Recall", threshold[2]], ["F1 Score", threshold[3]]]
                                    , columns=["Metric", "Threshold"]
                                    )

        output_df = pd.DataFrame(
            [["Accuracy", accuracy], ["Precision", precision], ["Recall", recall], ["F1 Score", f1_score]]
            , columns=["Metric", "Value"])

        output_df = pd.merge(output_df, threshold_df, how='left', on='Metric')

        output_df["Passed"] = output_df["Value"] >= output_df["Threshold"]

        # Return the final Dataframe
        return output_df

    else:
        print("Testing for Regression Metrics")
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)

        # Mean Squared Error - mean of the squared differences between the true and predicted labels
        mse = np.mean((true_values - predicted_values) ** 2)
        # -------------------------------------------------------------------

        # Root Mean Squared Error - square root of the MSE
        rmse = np.sqrt(mse)
        # -------------------------------------------------------------------

        # Mean Absolute Error - mean of the absolute differences between the true and predicted labels
        mae = np.mean(np.abs(true_values - predicted_values))
        # -------------------------------------------------------------------

        # Mean Absolute Percentage Error - mean of the absolute differences between the true and predicted labels,
        #                                  divided by the true labels, and multiplied by 100
        mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
        # -------------------------------------------------------------------

        # Map thresholds
        threshold_df = pd.DataFrame([["Mean Squared Error", threshold[0]], ["Root Mean Squared Error", threshold[1]]
                                    , ["Mean Absolute Error", threshold[2]]
                                    , ["Mean Absolute Percentage Error", threshold[3]]]
                                    , columns=["Metric", "Threshold"]
                                    )

        output_df = pd.DataFrame([["Mean Squared Error", mse], ["Root Mean Squared Error", rmse]
                                    , ["Mean Absolute Error", mae], ["Mean Absolute Percentage Error", mape]]
                                    , columns=["Metric", "Value"])

        output_df = pd.merge(output_df, threshold_df, how='left', on='Metric')

        output_df["Passed"] = output_df["Value"] <= output_df["Threshold"]

        # Return the final Dataframe
        return output_df
