# This python file contains the integration testing validation functions
import pandas as pd
import numpy as np


def find_outliers(data_df, categorical_threshold=0.1):
    """
    :param data_df: Pandas DataFrame Containing the Data in question
    :param categorical_threshold: The frequency threshold to mark a categorical value as an outlier
    :return: A Pandas DataFrame listing the index of all outlier values

    This function cycles through the column in the provided data and looks for outliers
    There are two checks that are preformed.
        1) Checking if a categorical column contains any values that fall below the acceptable frequency threshold and thus an outlier
        2) Checking if a numerical column contains any statistical outliers, identified using IQR values
    """

    outlier_df = pd.DataFrame(columns=["Column", "Value", "Index", "Issue"])
    for col in data_df.columns:
        if data_df[col].dtype == 'object' and all(isinstance(val, str) for val in data_df[col]):
            freq = data_df[col].value_counts(normalize=True)
            outlier_values = freq[freq < categorical_threshold].index
            outliers = data_df[data_df[col].isin(outlier_values)]
        elif data_df[col].dtype in ["float64","int64"]:
            q1 = data_df[col].quantile(0.25)
            q3 = data_df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = data_df[(data_df[col] < (q1 - 1.5 * iqr)) | (data_df[col] > (q3 + 1.5 * iqr))][col]
        else:
            pass

        # Prepare the temp "outliers" DF to be concatenated with the main outlier_df
        outliers = outliers.reset_index()
        outliers["Column"] = col
        outliers = outliers.rename(columns={"index":"Index",col:"Value"})
        outliers = outliers[["Column","Value","Index"]]
        outliers["Issue"] = "Outlier"
        outlier_df = pd.concat([outlier_df, outliers])

        return outlier_df


def find_duplicates(data_df):
    """
    :param data_df: Pandas DataFrame Containing the Data in question
    :return: A Pandas DataFrame listing the index of all duplicate values
    """

    duplicates = data_df[data_df.duplicated()]
    duplicates["Column"] = str(duplicates.columns.values)
    duplicates["Value"] = "N/A"
    duplicates = duplicates.reset_index()
    duplicates = duplicates.rename(columns={"index": "Index"})
    duplicates["Issue"] = "Duplicates"
    duplicates = duplicates[["Column", "Value", "Index", "Issue"]]

    return duplicates


def find_balance(data_df, label_field):
    """
    :param data_df: Pandas DataFrame Containing the Data in question
    :param label_field: String value of the label field
    :return: A Pandas DataFrame listing the class imbalance for each unique label
    """

    label_count = data_df.groupby(label_field).size().reset_index(name="label_count")
    total_count = label_count["label_count"].sum()
    label_count["label_percent"] = (label_count["label_count"]/total_count)*100

    label_count["imbalance_level"] = np.where(
        (label_count["label_percent"] > 20) & (label_count["label_percent"] <= 40)
        # A label being 20%-40$ of the data is mild imbalance
        , "Mild"
        , np.where(
            (label_count["label_percent"] > 1) & (label_count["label_percent"] <= 20)
            # A label being 1%-20% of the data is moderate imbalance
            , "Moderate"
            , np.where(
                (label_count["label_percent"] <= 1)
                # A label being <1% of the data is an extreme imbalance
                , "Extreme"
                # This is the 'else' for units 40%-100% of the data, which is not an imbalance
                , "None"
            )
        )
    )

    return label_count


def find_correlation(data_df, corr_method="pearson", threshold=0.75):
    """
    :param data_df: Pandas DataFrame Containing the Data in question
    :param corr_method: Correlation method to use when calculating the correlation among attributes in the DF
    :param threshold: The acceptable maximum correlation value allowed between two attributes
    :return: A Pandas DataFrame listing the correlation values for attribute pairs above the acceptable threshold
    """

    correlation = data_df.corr(method=corr_method)

    correlation = correlation.unstack().reset_index()

    correlation = correlation[correlation["level_0"] != correlation["level_1"]]

    correlation = correlation[correlation[0] >= threshold]

    correlation = correlation.rename(
        columns={"level_0": "Attribute_1", "level_1": "Attribute_2", 0: "Absolute_Correlation"})

    return correlation


def validate_dataset(data, categorical_threshold, label_field=None, corr_method="pearson", threshold=0.75,
                     data_type="excel"):
    """
    :param data: Pandas DataFrame Containing the Data in question OR a file path to an Excel or CSV file
    :param categorical_threshold: The frequency threshold to mark a categorical value as an outlier
    :param label_field: String value of the label field
    :param corr_method: Correlation method to use when calculating the correlation among attributes in the DF
    :param threshold: The acceptable maximum correlation value allowed between two attributes
    :param data_type: What form the "data" field is in
    :return: A dictionary of Pandas Dataframes where the key is the validation name and the value is the corresponding DF

    This is the wrapper function that a user can call to make use of all the integration validation checks provided by the library
    The supporting functions can be called individually if needed
    """

    # First step is to determine what was passed in and convert to a DF
    if data_type == "excel":
        data_df = pd.read_excel(data)
    elif data_type == "csv":
        data_df = pd.read_csv(data)
    elif data_type == "df":
        data_df = data.copy()
    else:
        return "Unexpected Type"

    # Check 1: Find Outliers
    outliers = find_outliers(data_df, categorical_threshold=categorical_threshold)

    # Check 2:
    duplicates = find_duplicates(data_df)

    # Check 3:
    if label_field is not None:
        balance = find_balance(data_df, label_field)

    # Check 4:
    correlation = find_correlation(data_df, corr_method=corr_method, threshold=threshold)

    if label_field is not None:
        return {"outliers": outliers, "duplicates": duplicates, "balance": balance, "correlation": correlation}
    else:
        return {"outliers": outliers, "duplicates": duplicates, "correlation": correlation}


def test_representative(dataset, sample):
    """
    :param dataset: Pandas DataFrame Containing the Data in question (entire population)
    :param sample: The sample of the population to test if representative of the entire population
    :return: A pandas DataFrame that details if any of the attributes in the sample Dataframe are not representative of the population
    """

    def calculate_ks(dataset, sample):
        # Calculate the empirical cumulative distribution functions (ECDFs) for the dataset and sample
        ecdf_dataset = np.cumsum(np.histogram(dataset, bins=1000, density=True)[0])
        ecdf_sample = np.cumsum(np.histogram(sample, bins=1000, density=True)[0])

        # Calculate the KS statistic
        ks_stat = np.max(np.abs(ecdf_dataset - ecdf_sample))

        # Calculate the p-value using the asymptotic formula
        n1 = len(dataset)
        n2 = len(sample)
        p_value = 2.0 * np.exp(-(2.0 * ks_stat ** 2) * n1 * n2 / (n1 + n2))

        return p_value

    return_df = pd.DataFrame(columns=["Feature", "P_Value", "Issue"])

    for feature in dataset.columns:
        p_value = calculate_ks(dataset[feature], sample[feature])
        # Distributions are not significantly different
        if p_value >= 0.05:
            return_df = pd.concat([return_df, pd.DataFrame([[feature, round(p_value, 6), "Representative"]],
                                                           columns=["Feature", "P_Value", "Issue"])])
        # Distributions are significantly different
        else:
            return_df = pd.concat([return_df,
                                   pd.DataFrame([[feature, round(p_value, 6), "Not Representative of Dataset"]],
                                                columns=["Feature", "P_Value", "Issue"])])

    return return_df
