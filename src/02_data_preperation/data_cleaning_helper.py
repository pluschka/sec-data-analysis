import pandas as pd
import numpy as np
import math
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample


def missing_strategy(df,
                     default_missing_strategy="delete",
                     except_replace_0=[],
                     except_replace_mean=[],
                     except_delete=[]):
    """
    Handles missing values according to the chosen strategy.

    default_missing_strategy (str) is applied to all variables unless they are
    listed in the exception lists:
        replace_0: replaces NAs with 0
        replace_mean: replaces NAs with the column mean
        delete: drops every row that contains an NA

    except_replace_0 (list): variables to replace with 0 instead of using the
    default strategy
    except_replace_mean (list): variables to replace with the mean instead of
    using the default strategy
    except_delete (list): variables for which rows with NAs should be dropped
    instead of using the default strategy

    Example:
        missing_strategy(
            diff_df,
            default_missing_strategy="delete",
            except_replace_0=["Variable_1"],
            except_replace_mean=["Variable_2"],
            except_delete=[],
        )

    Input:
        df (pd.DataFrame): our current data

    Output:
        New df (pd.DataFrame): dataset without NAs
    """
    for col in df.columns:
        if df[col].isna().any():
            if col in except_replace_0:
                df.loc[:, col] = df[col].fillna(0)
            elif col in except_replace_mean:
                df.loc[:, col] = df[col].fillna(df[col].mean())
            elif col in except_delete:
                df = df[~df[col].isna()]  # drop columns with na
            else:
                # default strategy
                if default_missing_strategy == "replace_0":
                    df.loc[:, col] = df[col].fillna(0)
                elif default_missing_strategy == "replace_mean":
                    df.loc[:, col] = df[col].fillna(df[col].mean())
                elif default_missing_strategy == "delete":
                    df = df[~df[col].isna()]  # drop columns with na
    return df


def plot_heatmap(df):
    """
    Plots a heatmap of the given df. Make sure to remove string variables.
    """
    corr_matrix = df.corr(numeric_only=False).abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.show()


def outlier_strategy(df,
                     default_outlier_strategy="delete",
                     except_replace_0=[],
                     except_replace_mean=[],
                     except_delete=[],
                     ignore=[]):
    """
    Removes outliers column-wise using the IQR rule and applies a chosen
    handling strategy.

    Outlier rule:
        For each numeric column, compute Q1 (25th pct), Q3 (75th pct),
        IQR = Q3 - Q1.
        Values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR are treated as outliers.

    Input:
        df (pd.DataFrame): input data. All columns are coerced to numeric
        (non-numeric → NaN) before processing.
        default_outlier_strategy (str): fallback strategy for columns not
        listed in the exception lists.
            Allowed: "delete" | "replace_0" | "replace_mean"
            - "delete": drop rows where the column has an outlier
            - "replace_0": replace outliers in the column with 0
            - "replace_mean": replace outliers in column with the column mean
        except_replace_0 (list[str]): columns for which outliers are replaced
        with 0 (overrides default)
        except_replace_mean (list[str]): columns for which outliers are
        replaced with the column mean (overrides default)
        except_delete (list[str]): columns for which rows with outliers are
        dropped (overrides default)
        ignore (list[str]): columns to skip entirely

    Output:
        pd.DataFrame: DataFrame after outlier handling.

    Notes:
        - Only numeric columns are processed;
        coercion uses pd.to_numeric(..., errors="coerce").
        - If IQR is 0, no values will be marked as outliers for that column
        (bounds collapse to Q1=Q3).
    """
    cols = df.columns
    df = df[cols].apply(pd.to_numeric, errors='coerce')

    for cols in df.select_dtypes(include=['number']).columns:
        if cols in ignore:
            continue
        # calculate Q1, Q3 und IQR
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1

        # define threshold for outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if cols in except_replace_0:
            df.loc[:, cols] = df[cols].apply(
                lambda x: 0 if x < lower_bound or x > upper_bound else x)
        elif cols in except_replace_mean:
            mean_val = df[cols].mean()
            df.loc[:, cols] = df[cols].apply(
                lambda x: mean_val if x < lower_bound or
                x > upper_bound else x)
        elif cols in except_delete:
            df = df[~((df[cols] < lower_bound) | (df[cols] > upper_bound))]
        else:
            # strategies
            if default_outlier_strategy == "replace_0":
                df[cols] = df[cols].apply(
                    lambda x: 0 if x < lower_bound or x > upper_bound else x)
            elif default_outlier_strategy == "replace_mean":
                mean_val = df[cols].mean()
                df[cols] = df[cols].apply(
                    lambda x: mean_val if x < lower_bound or
                    x > upper_bound else x)
            elif default_outlier_strategy == "delete":
                df = df[~((df[cols] < lower_bound) | (df[cols] > upper_bound))]
    return df


def skewness_overview(df):
    """
    Returns, for each numeric column, the original skewness of the variable
    and the skewness after the following transformations:

    Transformations considered:
    - log(x)
    - log1p(x)
    - sqrt(x)
    - boxcox(x)
    - -log1p(x)
    - log(max - x + 1)
    - power_transform (Yeo-Johnson)

    Note: log(x) and boxcox(x) are not applicable to negative values.

    If transformation is not possible (e.g., for binary variables), print -

    Input:
        df (pd.DataFrame): our current DataFrame

    Output:
        pd.DataFrame: summary with skewness values per transformation
    """

    results = []

    # skip binary variables
    for col in df.columns:
        if df[col].dtype == bool:
            results.append({
                "row": col,
                "original": "-",
                "log(x)": "-",
                "log1p": "-",
                "sqrt": "-",
                "boxcox": "-",
                "power_transform": "-"

            })
            continue

        series = pd.to_numeric(df[col], errors='coerce')
        row = {"Row": col}

        # original
        try:
            # use only useful variables
            # valid_series = series.dropna()

            # condition: enough distinct values
            if series.nunique() <= 1:
                row["original"] = "not calculable"
            elif series.var() < 1e-8:
                row["original"] = "almost constant"
            else:
                row["original"] = round(skew(series), 3)
        except Exception:
            row["original"] = "-"

        # log(x) only für x > 0
        try:
            if (series <= 0).any():
                row["log(x)"] = "not applicable because 0 or negative value"
            else:
                transformed = np.log(series)
                if len(transformed) <= 1:
                    row["log(x)"] = "not calculable"
                elif np.var(transformed) < 1e-8:
                    row["log(x)"] = "almost constant"
                else:
                    row["log(x)"] = round(skew(transformed), 3)
        except Exception:
            row["log(x)"] = "-"

        # log1p(x) only für x > -1
        try:
            if (series <= -1).any():
                row["log1p"] = "not calculable because values smaller then -1"
            else:
                transformed = np.log1p(series)
                row["log1p"] = round(skew(transformed), 3)
        except Exception:
            row["log1p"] = "-"

        # sqrt(x) only für x ≥ 0
        try:
            if (series < 0).any():
                row["sqrt"] = "not calculable because of values smaller then 0"
            else:
                transformed = np.sqrt(series)
                row["sqrt"] = round(skew(transformed), 3)
        except Exception:
            row["sqrt"] = "-"

        # boxcox(x) only für x > 0
        try:
            if (series <= 0).any():
                row["boxcox"] = "not calculable because values smaller then 0"
            else:
                transformed, _ = boxcox(series)
                row["boxcox"] = round(skew(transformed), 3)
        except Exception:
            row["boxcox"] = "-"

        # power_transform (Yeo-Johnson)
        try:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            transformed = pt.fit_transform(series.values.reshape(-1, 1))\
                            .flatten()
            row["power_transform"] = round(skew(transformed), 3)
        except Exception:
            row["power_transform"] = "-"

        results.append(row)
    return pd.DataFrame(results)


def balance(df, name_target="target"):

    df_majority = df[~df[name_target]]
    df_minority = df[df[name_target]]

    # random sample of majority class same size as majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # without returning
                                       n_samples=len(df_minority),
                                       random_state=420)  # set seed

    # union data
    df = pd.concat([df_majority_downsampled, df_minority])

    diagram(df,
            diagram="histplot",
            variable="target",
            name_target=None,
            save_as=None)
    return df


def scatterplot(df, variable=None, name_target="target"):
    """
    Creates scatter plots for every combination of the provided variables and
    saves them together as a grid.

    Input:
        df (pd.DataFrame): our current data
        variable (list or None): list with two variable names for a single
        scatter plot; if None, plot all pairwise combinations
        name_target (str): target variable used for color mapping (hue)
    """

    # plots for chosen variables
    if variable is not None:
        x, y = variable
        sns.scatterplot(data=df,
                        x=x,
                        y=y,
                        hue=name_target if name_target in df.columns else None)
        plt.tight_layout()
        plt.savefig(f"scatter_{x}_and_{y}.png", dpi=300)
        plt.close()
        return

    # plots for all variables
    variable = df.select_dtypes(include='number').columns.tolist()
    combination = list(itertools.combinations(variable, 2))

    for v in variable:
        relevant_combination = [(x_var, y_var) for x_var, y_var
                                in combination if v in (x_var, y_var)]

        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        axes = axes.flatten()

        for idx, (x_var, y_var) in enumerate(relevant_combination):
            if idx >= len(axes):
                break
            sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                hue=name_target if name_target in df.columns else None,
                ax=axes[idx]
            )

        # remove irrelevant axe
        for ax in axes[len(relevant_combination):]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(f"Scatter_{v}.png", dpi=300)
        plt.close()


def diagram(df,
            diagram="boxplot",
            variable=None,
            name_target="target",
            title=None):
    """
    Creates boxplot or histograms for each variable in the DataFrame, or for a
    specific list of variables passed via `variable`. If `name_target` is
    provided, plots are stratified by that target if `name_target=None`,
    the overall distribution is shown.

    Input:
        df (pd.DataFrame): the current dataset
        diagram (str): chart type; one of {"boxplot", "histplot"}
        variable (list[str] or None): variables to plot; e.g., ["Variable_1"].
            If None (default), plot all eligible variables.
        name_target (str or None): target column used to split the plots by
        class (hue).
            If None, no stratification is applied.
        save_as (str or None): filename for saving the PNG output; if None, do
        not save

    Output:
        Plots: histogram(s)/boxplot(s) for the requested variable(s),
        optionally split by `name_target`
    """

    # plot for one  variable
    if variable is not None:
        plt.figure(figsize=(6, 4))
        if diagram == "boxplot":
            if name_target is None:
                ax = sns.boxplot(y=df[variable])
            else:
                ax = sns.boxplot(data=df, x=name_target, y=variable)
        elif diagram == "histplot":
            if name_target is None:
                ax = sns.histplot(data=df, x=variable, bins=30, kde=False)
            else:
                ax = sns.histplot(data=df,
                                  x=variable,
                                  hue=name_target,
                                  bins=30,
                                  kde=False)
        title_text = f"{variable}" if title is None else\
                     f"{variable} - {title}"
        ax.set_title(title_text)

        plt.tight_layout()
        plt.show()
        return

    # plots for all variables
    else:

        variable = df.select_dtypes(include='number').columns.tolist()
        if name_target in variable:
            variable.remove(name_target)

        count = len(variable)
        columns = 3  # count rows in grid
        rows = math.ceil(count / columns)

        fig, axes = plt.subplots(rows,
                                 columns,
                                 figsize=(columns * 5, rows * 4))
        axes = axes.flatten()  # makes 2D array to a 1D list

        for i, var in enumerate(variable):
            ax = axes[i]
            if diagram == "boxplot":
                if name_target is None:
                    sns.boxplot(y=df[var], ax=ax)
                else:
                    sns.boxplot(data=df, x=name_target, y=var, ax=ax)

            elif diagram == "histplot":
                if name_target is None:
                    sns.histplot(data=df, x=var, bins=30, kde=False, ax=ax)
                else:
                    sns.histplot(data=df,
                                 x=var,
                                 hue=name_target,
                                 bins=30,
                                 kde=False,
                                 ax=ax)

            # title for each subplot
            if title:
                ax.set_title(f"{var} - {title}")

        # hide empty axe
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
    plt.show()
