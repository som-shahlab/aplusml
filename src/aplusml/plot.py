"""Plotting functions"""
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.calibration
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from plotnine import *
import warnings

plt.style.use('ggplot')
warnings.filterwarnings( "ignore", module = "plotnine\..*" )

################################################
# Simulation analysis
################################################
def plot_mean_utility_v_threshold(title: str, 
                                  df: pd.DataFrame,
                                  label_sort_order: list[str] = None,
                                  label_names: list[str] = None,
                                  label_title: str = '') -> ggplot:
    """_summary_

    Args:
        title (str): _description_
        df (pd.DataFrame): Output of `run.run_test()`
                            Columns: threshold, mean_utility, std_utility, sem_utility, mean_work_per_timestep, label
        label_names (list, optional): If specified, actual names displayed for each label
        label_title (str, optional): If specified, title of label legend

    Returns:
        ggplot: _description_
    """    
    df = df.copy()
    df['label'].fillna('N/A', inplace=True)
    df['label'] = pd.Categorical(df['label'], categories=label_sort_order if label_sort_order is not None else pd.unique(df['label']))
    if not label_names: label_names = df['label'].unique()
    # Reorder labels ordered from lowest to highest max(mean_utility))
    p = (
        ggplot(df, aes(y='mean_utility', x='threshold', color='label')) +
        geom_ribbon(aes(ymin='mean_utility - sem_utility', 
                               ymax='mean_utility + sem_utility',
                               fill='label'), 
                           alpha=0.3,
                           outline_type = None,
                           color=None) +
        geom_point(size=0.5, show_legend=False) +
        labs(x="Model Cutoff Threshold",
             y="Achieved Utility Per Patient",
             title=f"{title}") +
        scale_fill_discrete(name = label_title or 'Setting', labels=label_names)
    )
    return p

def plot_dodged_bar_mean_utilities(title: str, 
                                   df: pd.DataFrame,
                                   label_sort_order: list[str] = None,
                                   label_names: list[str] = None,
                                   color_sort_order: list[str] = None,
                                   color_names: list[str] = None,
                                   is_percent_of_optimistic: bool = False,
                                   x_label: str = None) -> ggplot:
    """Plot Relative Utility using the Optimistic model as a baseline
    (even though Relative Utility is typically measured as a fraction of a Perfect classifier)
    Plot:
        'label' = x-axis,
        'color' = primary color of the line 

    Args:
        title (str): _description_
        df (pd.DataFrame): Must contain 3 columns: y, label, color
        label_sort_order (list, optional): If specified, determines ordering of labels in x-axis. Defaults to None.
        label_names (list, optional): If specified, actual names displayed for each label
        color_sort_order (list, optional): If specified, determines ordering of colored bars. Defaults to None.
        color_names (list, optional): If specified, actual names displayed for each color
        is_percent_of_optimistic (bool, optional): If TRUE, then measure everything in terms of the % of the max optimistic setting. Defaults to False.
    """    
    df = df.copy()
    # Sort utilities
    baseline = df['y'].min()
    if is_percent_of_optimistic:
        # We scale everything by the 'optimistic' pathway that yields the max utility (since there may be multiple 'optimistic' rows)
        utility_optimistic = df[df['label'] == 'optimistic']['y'].max()
        df['y'] = (df['y'] - baseline) / (utility_optimistic - baseline)
    else:
        df['y'] = df['y'] - baseline
    df = df.sort_values('y')
    # NOTE: need to `fillna` b/c Pandas doesn't let you set None as a categorical value
    df['label'].fillna('N/A', inplace=True)
    df['label'] = pd.Categorical(df['label'], categories=label_sort_order if label_sort_order is not None else pd.unique(df['label']))
    if not label_names: label_names = df['label'].unique()
    df['color'].fillna('N/A', inplace=True)
    df['color'] = pd.Categorical(df['color'], categories=color_sort_order if color_sort_order is not None else pd.unique(df['color']))
    if not color_names: color_names = df['color'].unique()
    # Make plot
    p = (
        ggplot(df, aes(x = 'label', y = 'y', fill = 'color')) + 
        geom_col(stat='identity', position='dodge') +
        labs(x = f"{title}" if not x_label else x_label,
             y = f"Achieved Utility Per Patient Over 'Treat None'\n { 'as Fraction of Optimistic Pathway Utility' if is_percent_of_optimistic else ''}",
             title = f"{title}",
             color='') + 
        scale_fill_discrete(name = "Model", labels=color_names)
    )
    return p

def plot_line_mean_utilities(title: str, 
                                df: pd.DataFrame,
                                group_sort_order: list = None,
                                groups_to_drop: list = None,
                                label_sort_order: list = None,
                                color_sort_order: list = None,
                                color_names: list[str] = None,
                                shape_sort_order: list = None,
                                is_percent_of_optimistic: bool = False,
                                color_title: str = None,
                                shape_title: str = None,
                                x_label: str = None) -> ggplot:
                                   
    """Plot Relative Utility using the Optimistic model as a baseline
    (even though Relative Utility is typically measured as a fraction of a Perfect classifier)
    This uses different shapes of the same color to group lines belonging to the same category
    Unlike dodged bar, this supports multiple groupings
    Plot:
        'label' = x-axis,
        'group' = groups together points on the same line 
        'color' = primary color of the line 
        'shape' = shape of the points on a line (e.g. circle, triangle)

    Args:
        title (str): _description_
        df (pd.DataFrame): Must contain 3 columns: y, label, group (optional: color, shape)
        label_sort_order (list, optional): If specified, determines ordering of x in x-axis. Defaults to None.
        color_sort_order (list, optional): If specified, determines ordering of coloring. Defaults to None.
        color_names (list, optional): If specified, actual names displayed for each color
        shape_sort_order (list, optional): If specified, determines ordering of shapes. Defaults to None.
        is_percent_of_optimistic (bool, optional): If TRUE, then measure everything in terms of the % of the max optimistic setting. Defaults to False.
    """    
    df = df.copy()
    # Sort utilities
    baseline = df['y'].min()
    if is_percent_of_optimistic:
        # We scale everything by the 'optimistic' pathway that yields the max utility (since there may be multiple 'optimistic' rows)
        utility_optimistic = df[df['label'] == 'optimistic']['y'].max()
        df['y'] = 100*(df['y'] - baseline) / (utility_optimistic - baseline)
    else:
        df['y'] = df['y'] - baseline
    df = df.sort_values('y')
    # NOTE: need to `fillna` b/c Pandas doesn't let you set None as a categorical value
    df['label'] = pd.Categorical(df['label'], categories=label_sort_order if label_sort_order is not None else pd.unique(df['label']))
    df['group'].fillna('N/A', inplace=True)
    df['group'] = pd.Categorical(df['group'], categories=group_sort_order if group_sort_order is not None else pd.unique(df['group']))
    if 'color' in df:
        df['color'].fillna('N/A', inplace=True)
        df['color'] = pd.Categorical(df['color'], categories=color_sort_order if color_sort_order is not None else pd.unique(df['color']))
    if 'shape' in df:
        df['shape'].fillna('N/A', inplace=True)
        df['shape'] = pd.Categorical(df['shape'], categories=shape_sort_order if shape_sort_order is not None else pd.unique(df['shape']))
    # Drop certain groups (if specified)
    if groups_to_drop is not None:
        for g in groups_to_drop:
            df = df[df['group'] != g]
    if not color_names: color_names = df['color'].unique()
    # Make plot
    p = (
        ggplot(df, aes(**{
            'x': 'label',
            'y': 'y',
            'group': 'group',
            **({ 'color': 'color' } if 'color' in df else {}),
            **({ 'shape': 'shape' } if 'shape' in df else {}),
        })) +
        geom_point(size=2) +
        geom_line(size=0.5) + 
        labs(x = f"{title}" if not x_label else x_label,
             y = f"Achieved Utility Over Baseline { ' v. Optimistic (%)' if is_percent_of_optimistic else ''}",
             title = f"{title}",
             color=color_title if color_title else '',
             shape=shape_title if shape_title else '') +
        scale_color_hue(name = "Model", labels = color_names)
    )
    return p

def plot_bar_mean_utilities(title: str, 
                            plot_avg_utilities: dict[float]) -> ggplot:
    # Sort utilities
    sorted_plot_avg_utilities = sorted(list(plot_avg_utilities.items()), key = lambda x: x[1])
    labels = [x[0] for x in sorted_plot_avg_utilities]
    values = np.array([x[1] for x in sorted_plot_avg_utilities])
    baseline = np.min(list(plot_avg_utilities.values()))
    percents = 100*(values - baseline) / (plot_avg_utilities['optimistic'] - baseline)
    # Dataframe
    df = pd.DataFrame()
    df['labels'] = labels
    df['values'] = list(values)
    df['percents'] = list(percents)
    df['labels'] = pd.Categorical(df['labels'], categories=pd.unique(df['labels']))
    df = df.sort_values('percents')
    # Make plot
    p = (
        ggplot(df, aes(x = 'labels', y = 'percents')) + 
        geom_bar(stat='identity', fill="lightblue") +
        labs(x = f"{title}",
             y = "Achieved Relative Utility Over Baseline v. Optimistic (%)",
             title = f"Impact of {title}")
    )
    return p

def plot_line_compare_multiple_settings(title: str,
                                        df: list[pd.DataFrame],
                                        x_label: str = None,
                                        line_labels: list[str] = None) -> ggplot:
    """Plots multiple lines on the same x- and y-axis
        Used to generate Figure 2 from Jung et al. 2021

    Args:
        title (str): _description_
        df (pd.DataFrame): Columns: x, y, line

    Returns:
        ggplot: _description_
    """
    p = (
        ggplot(df, aes(x = 'x', y = 'y', color = 'line', group = 'line')) + 
        geom_point(size = 1) +
        geom_line(size = 1) + 
        (scale_color_hue(labels=line_labels) if line_labels else ()) + 
        labs(x = f"{x_label if x_label else ''}",
             y = "Change in Achieved Utility v. Baseline",
             title = f"{title}",
             color = '') +
        theme(
            legend_direction='vertical',
            legend_position = (0.75, 0.75),
            legend_background=element_rect(fill='white'),
            legend_title=element_blank(),
        )
    )
    return p
    
################################################
# Theoretical Utility Analysis
################################################
"""Pattern for all plots:
    Args:
        df_preds (pd.DataFrame): DataFrame that must contain two columns -- 'y' and 'y_hat'
        utilities (dict): Dictionary that must contain four keys: 'tp', 'fp', 'tn', 'fn'

    Returns:
        Plot object
"""

PADDING = 0.05 # How much padding to add to plots that get bounded by (0,1) - e.g. ROC Curve
def get_df_utility_from_df_preds(df_preds: pd.DataFrame, 
                                 utilities: dict, 
                                 thresholds: np.ndarray,
                                 is_add_0_and_1: bool = True
                                 ) -> pd.DataFrame:
    """Adds metrics (Utilities + TP/FP/TN/FN) to DataFrame of predictions

    Args:
        thresholds (np.ndarray): Array of model thresholds to consider
        is_add_0_and_1 (bool): If TRUE, add 0 and 1 to the start/end of the `thresholds` array

    Returns:
        pd.DataFrame: DataFrame that mirrors "df_preds", but with additional columns
    """
    utility_rows = []
    thresholds = np.sort(np.array(list(set(thresholds).union(set([0,1] if is_add_0_and_1 else [])))))
    for t in thresholds:
        preds = df_preds['y_hat'] >= t
        tp = np.sum((df_preds['y'] == preds).values & (preds == 1).values)
        fp = np.sum((df_preds['y'] != preds).values & (preds == 1).values)
        tn = np.sum((df_preds['y'] == preds).values & (preds == 0).values)
        fn = np.sum((df_preds['y'] != preds).values & (preds == 0).values)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        utility_rows.append({
            'threshold' : t,
            'prevalence' : (tp + fn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0, # Proportion of y's that are TRUE
            'ppv' : ppv, # Same as precision
            'precision' : ppv, # Same as PPV
            'tpr' : tpr, # Same as recall
            'recall' : tpr, # Same as TPR
            'fpr' : fpr,
            'tp' : tp,
            'fp' : fp,
            'tn' : tn,
            'fn' : fn,
            'total' : tp + fp + tn + fn,
            'work' : tp + fp,
            'unit_utility' : (utilities['tp'] * tp + utilities['fp'] * fp + utilities['tn'] * tn + utilities['fn'] * fn) / (tp + fp + tn + fn),
        })
    df_utility = pd.DataFrame(utility_rows)
    return df_utility

def plot_hist_pred(df_preds: pd.DataFrame, ax: plt.Axes = None) -> plt.figure:
    """Generate Histogram of predictions
        x-axis = predictions ('y_hat')
        y-axis = density
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)

    # Plot
    ax.hist(df_preds['y_hat'].values, bins=50)
    ax.set_title(f"Histogram of Predictions", fontdict={'fontsize' : 12})
    ax.set_ylabel("Count", fontdict={'fontsize' : 10})
    ax.set_xlabel("Prediction", fontdict={'fontsize' : 10})
    return fig

def calc_pearsonr(a, b):
    """Pearson correlation coefficient between two 1d vectors"""
    return np.corrcoef(a, b)[0,1]

def calc_spearmanr(a, b):
    """Spearman correlation coefficient between two 1d vectors"""
    return np.corrcoef(a, b)[0,1]

def plot_pred_v_true(df_preds: pd.DataFrame, ax: plt.Axes = None) -> plt.figure:
    """Generate Predicted v. True Values
        x-axis = predictions ('y_hat')
        y-axis = ground truth ('y')
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)

    # Calculate Pearson Correlation
    pearson_corr = calc_pearsonr(df_preds['y_hat'], df_preds['y'])
    spearman_corr = calc_spearmanr(df_preds['y_hat'], df_preds['y'])
    # Plot
    # Source: https://stackoverflow.com/a/20107592/3015186
    x = df_preds['y_hat'].values
    y = df_preds['y'].values
    xy = np.vstack([x, y])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(xy.T)
    z = np.exp(kde.score_samples(xy.T))
    density = ax.scatter(x,y, c=z, s=10)

    ax.set_title(f"Predictions v. Ground Truth\nPearson $r$={round(pearson_corr, 3)}, Spearman $\\rho$={round(spearman_corr, 3)}", fontdict={'fontsize' : 12})
    ax.set_ylabel("True Label", fontdict={'fontsize' : 10})
    ax.set_xlabel("Prediction", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_calibration_curve(df_preds: pd.DataFrame, ax: plt.Axes = None) -> plt.figure:
    """Generate Calibration curve
        x-axis = predictions ('y_hat')
        y-axis = ground truth ('y')
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    x = np.linspace(0,1,1000)

    # Calibration Curve
    #   Calculate
    fop, mpv = sklearn.calibration.calibration_curve(df_preds['y'], df_preds['y_hat'], n_bins=10)
    #   Plot
    ax.plot(mpv, fop, '-bo', label='Model')
    ax.plot(x, x, linestyle='--', label='Perfect')

    ax.legend()
    ax.set_title(f"Calibration curve", fontdict={'fontsize' : 12})
    ax.set_ylabel("Fraction of Positives", fontdict={'fontsize' : 10})
    ax.set_xlabel("Predicted Probability of Positive", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_confusion_matrix(df_preds: pd.DataFrame, utilities: dict = None, threshold: float = None, ax: plt.Axes = None) -> plt.figure:
    """Generate Confusion Matrix @ highest utility threshold (if `threshold` is not specified)
        x-axis = predictions ('y_hat')
        y-axis = ground truth ('y')
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=True)
    max_utility_idx = df_utility['unit_utility'].argmax()
    threshold = df_utility['threshold'][max_utility_idx]

    # Plot
    cm = confusion_matrix(df_preds['y'], df_preds['y_hat'] >= threshold)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    if ax:
        disp.plot(ax = ax)
    else:
        fig = disp.figure_
        ax = disp.ax_
    ax.set_title(f"Confusion Matrix\nUsing Cutoff Threshold $t$ with Max Unit Utility\n($t$ = {round(threshold, 3)})", fontdict={'fontsize' : 12})
    ax.set_ylabel("True Label", fontdict={'fontsize' : 10})
    ax.set_xlabel("Predicted Label", fontdict={'fontsize' : 10})
    ax.grid(False)
    return fig

def plot_roc_curve(df_preds: pd.DataFrame, utilities: dict = None, ax: plt.Axes = None) -> plt.figure:
    """Generate ROC curve
            Explanation of indifference curves: http://www0.cs.ucl.ac.uk/staff/ucacbbl/roc/
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    x = np.linspace(0,1,1000)

    # ROC Curve
    #   Calculate
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(df_preds['y'], df_preds['y_hat'])
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    #   Plot
    ax.plot(fpr, tpr, 'b', label='ROC Curve')
    ax.plot(x, x, 'k--', label='Random')

    # Utility
    if utilities:
        #   Calculate
        df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds)
        max_utility_idx = df_utility['unit_utility'].argmax()
        prevalence = df_preds['y'].mean() # Proportion of positive cases
        indifference_curve_slope = (1-prevalence) / prevalence * (utilities['tn'] - utilities['fp']) / (utilities['tp'] - utilities['fn'])
        #   Plot
        #       Max utility
        max_utility_x = df_utility.iloc[max_utility_idx]['fpr']
        max_utility_y = df_utility.iloc[max_utility_idx]['tpr']
        ax.plot(max_utility_x, max_utility_y, 'bo')
        ax.plot(x, indifference_curve_slope * x + (max_utility_y - indifference_curve_slope * max_utility_x), 'b--', label='Max Utility')
        #       Treat All
        all_x = df_utility.iloc[0]['fpr']
        all_y = df_utility.iloc[0]['tpr']
        ax.plot(all_x, all_y, 'ro')
        ax.plot(x, indifference_curve_slope * x + (all_y - indifference_curve_slope * all_x), 'r--', label='Treat All Utility')
        #       Treat None
        none_x = df_utility.iloc[-1]['fpr']
        none_y = df_utility.iloc[-1]['tpr']
        ax.plot(none_x, none_y, 'go')
        ax.plot(x, indifference_curve_slope * x + (none_y - indifference_curve_slope * none_x), 'g--', label='Treat None Utility')
    ax.legend()
    ax.set_title(f"ROC curve{' with indifference curves' if utilities else ''}\nAUC={auc}", fontdict={'fontsize' : 12})
    ax.set_ylabel("TPR", fontdict={'fontsize' : 10})
    ax.set_xlabel("FPR", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_precision_recall_curve(df_preds: pd.DataFrame, utilities: dict = None, ax: plt.Axes = None) -> plt.figure:
    """Generate Precision-Recall curve
        Explanation of Expected Utility (EU) calculations (Appendix A): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2804257/pdf/nihms-108324.pdf
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    x = np.linspace(0,1,1000)

    # ROC Curve
    #   Calculate
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(df_preds['y'], df_preds['y_hat'])
    auc = round(sklearn.metrics.auc(recall, precision), 3)
    prevalence = df_preds['y'].mean() # Proportion of positive cases
    #   Plot
    ax.plot(recall, precision, 'b', label='PR Curve')
    ax.plot(x, [prevalence] * 1000, 'k--', label='Random')

    # Utility
    if utilities:
        #   Calculate
        df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds)
        max_utility_idx = df_utility['unit_utility'].argmax()
        #   Generate EU contours for plot
        utility_contour_x, utility_contour_y = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
        with np.errstate(divide='ignore',invalid='ignore'): # Ignore divide by 0
            expected_utility = (prevalence * utility_contour_x * (utilities['fp'] - utilities['tn'])) / utility_contour_y + utilities['tn'] + prevalence * (utilities['fn'] - utilities['tn'] + utility_contour_x * (utilities['tn'] + utilities['tp'] - utilities['fp'] - utilities['fn'])) 
        #   Max utility
        max_utility_x = df_utility.iloc[max_utility_idx]['recall'] # TPR
        max_utility_y = df_utility.iloc[max_utility_idx]['precision'] # PPV
        ax.plot(max_utility_x, max_utility_y, 'bo')
        #   Utility contour
        cs = ax.contour(utility_contour_x, utility_contour_y, expected_utility)
        ax.clabel(cs, inline=1, fontsize=8)
        # for i in range(len(cs.levels)):
        #     cs.collections[i].set_label(cs.levels[i])
    ax.legend()
    ax.set_title(f"Precision-Recall curve{' with indifference curves ' if utilities else ''}\nAUC={auc}", fontdict={'fontsize' : 12})
    ax.set_ylabel("Precision (PPV)", fontdict={'fontsize' : 10})
    ax.set_xlabel("Recall (TPR)", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_ppv_v_utility_curve(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate PPV v. Unit Utility curve
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=False)
    max_utility_idx = df_utility['unit_utility'].argmax()

    # Plot
    #   Curve
    ax.plot(df_utility['ppv'], df_utility['unit_utility'], 'k', label='Utility Curve')
    #   Max utility
    max_utility_x = df_utility.iloc[max_utility_idx]['ppv']
    max_utility_y = df_utility.iloc[max_utility_idx]['unit_utility']
    ax.plot(max_utility_x, max_utility_y, 'bo')
    ax.legend()
    ax.set_title(f"PPV v. Unit Utility", fontdict={'fontsize' : 12})
    ax.set_ylabel("Unit Utility", fontdict={'fontsize' : 10})
    ax.set_xlabel("Precision (PPV)", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    return fig

def plot_threshold_v_utility_curve(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate Cutoff Threshold v. Unit Utility curve
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=True)
    max_utility_idx = df_utility['unit_utility'].argmax()

    # Plot
    #   Curve
    ax.plot(df_utility['threshold'], df_utility['unit_utility'], 'k', label='Utility Curve')
    #   Max utility
    max_utility_x = df_utility.iloc[max_utility_idx]['threshold']
    max_utility_y = df_utility.iloc[max_utility_idx]['unit_utility']
    ax.plot(max_utility_x, max_utility_y, 'bo')
    ax.legend()
    ax.set_title(f"Cutoff Threshold v. Unit Utility", fontdict={'fontsize' : 12})
    ax.set_ylabel("Unit Utility", fontdict={'fontsize' : 10})
    ax.set_xlabel("Cutoff Threshold", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    return fig

def plot_work_v_utility(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate Work v. Unit Utility curve
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=True)
    max_utility_idx = df_utility['unit_utility'].argmax()

    # Plot
    #   Curve
    ax.plot(df_utility['work'] / df_utility['total'], df_utility['unit_utility'], 'k', label='Utility Curve')
    #   Max utility
    max_utility_x = df_utility.iloc[max_utility_idx]['work'] / df_utility.iloc[max_utility_idx]['total']
    max_utility_y = df_utility.iloc[max_utility_idx]['unit_utility']
    ax.plot(max_utility_x, max_utility_y, 'bo')
    ax.legend()
    ax.set_title(f"Work v. Unit Utility", fontdict={'fontsize' : 12})
    ax.set_ylabel("Unit Utility", fontdict={'fontsize' : 10})
    ax.set_xlabel("% Positive Predictions", fontdict={'fontsize' : 10})
    ax.set_xlim(0 - PADDING, 1 + PADDING)
    return fig

def plot_work_v_ppv_tpr_fpr(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate Work v. PPV/TPR/FPR curve
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=False)

    # Plot
    #   PPV
    ax.plot(df_utility['work'] / df_utility['total'], df_utility['ppv'], 'r', label='PPV')
    #   TPR
    ax.plot(df_utility['work'] / df_utility['total'], df_utility['tpr'], 'g', label='TPR')
    #   FPR
    ax.plot(df_utility['work'] / df_utility['total'], df_utility['fpr'], 'b', label='FPR')
    ax.legend()
    ax.set_title(f"Work v. PPV/TPR/FPR", fontdict={'fontsize' : 12})
    ax.set_ylabel("Value", fontdict={'fontsize' : 10})
    ax.set_xlabel("% Positive Predictions", fontdict={'fontsize' : 10})
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_threshold_v_ppv_tpr_work(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate Cutoff Threshold v. PPV/TPR/Work curve
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=False)

    # Plot
    #   PPV
    ax.plot(df_utility['threshold'], df_utility['ppv'], 'r', label='PPV')
    #   TPR
    ax.plot(df_utility['threshold'], df_utility['tpr'], 'g', label='TPR')
    #   Fraction Positive (Work / Total)
    ax.plot(df_utility['threshold'], df_utility['work'] / df_utility['total'], 'b', label='% Positive Predictions')
    ax.legend()
    ax.set_title(f"Cutoff Threshold v. PPV/TPR/Work", fontdict={'fontsize' : 12})
    ax.set_ylabel("Value", fontdict={'fontsize' : 10})
    ax.set_xlabel("Cutoff Threshold", fontdict={'fontsize' : 10})
    ax.set_ylim(0 - PADDING, 1 + PADDING)
    return fig

def plot_decision_curve(df_preds: pd.DataFrame, utilities: dict, ax: plt.Axes = None) -> plt.figure:
    """Generate Decision Curve for model
    Risk Threshold (R) is defined as "the cutpoint for calling a result positive that maximizes expected utility"
    Net Benefit of X =>
        y-axis = X - U_none (i.e. everything is realtive to Treat None)
        x-axis = R
    Thus, "Treat None" on this graph = U_none - U_none = 0
    Thus, at R = r, the "Treat All" graph is saying what the net benefit would be if you treated everyone AND the optimal risk threshold = r
        Guarantee: "Treat All" and "Treat None" lines should intersect at R = Prevalence
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # Calculate
    # NOTE: 'thresholds' may be a diff size than df_utility['threshold'] b/c we add 0,1 to it in 'get_df_utility_from_df_preds'
    thresholds = np.sort(df_preds['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(df_preds, utilities, thresholds, is_add_0_and_1=False)
    prevalence = df_preds['y'].mean() # Proportion of positive cases

    # Plot
    # Treat All
    # Explanation of how to calculate net benefits (Page 8):
    #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2804257/pdf/nihms-108324.pdf
    # Explanation of risk thresholds / curves (Page 5): 
    #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2577036/pdf/nihms74373.pdf
    net_benefit_all = (prevalence - (1 - prevalence) *  df_utility['threshold'] / (1 - df_utility['threshold']) )
    ax.plot(df_utility['threshold'], net_benefit_all, 'r--', label='Treat All')
    # Treat None
    net_benefit_none = [0] * df_utility['threshold'].shape[0]
    ax.plot(df_utility['threshold'], net_benefit_none, 'g--', label='Treat None')
    # Model
    net_benefit_model = (prevalence * df_utility['tpr'] - (1 - prevalence) * df_utility['fpr'] * df_utility['threshold'] / (1 - df_utility['threshold']) )
    ax.plot(df_utility['threshold'], net_benefit_model, 'k', label='Model')
    ax.legend()
    ax.set_title(f"Decision Curve", fontdict={'fontsize' : 12})
    ax.set_ylabel("Net Benefit", fontdict={'fontsize' : 10})
    ax.set_xlabel("Risk Threshold", fontdict={'fontsize' : 10})
    ax.set_xlim(0, 1)
    ax.set_ylim(-np.maximum(net_benefit_all.max(), net_benefit_model.max()),
                np.maximum(net_benefit_all.max(), net_benefit_model.max()))
    return fig

def plot_relative_utility_curve(lines: list[dict[pd.DataFrame]],
                                utilities: dict,
                                xlim: Tuple = (0, 1),
                                ylim: Tuple = (-0.1, None),
                                ax: plt.Axes = None) -> plt.figure:
    """Generate Relative Utility Curve
        "The relative utility curve plots the fraction of the expected utility 
        of perfect prediction obtained by the risk prediction model at the 
        optimal cut point associated with the risk threshold R"
    """
    fig, ax = plt.subplots() if ax is None else (None, ax)
    # NOTE: 'thresholds' may be a diff size than df_utility['threshold'] b/c we add 0,1 to it in 'get_df_utility_from_df_preds'
    thresholds = np.sort(lines[0]['df']['y_hat'].unique())[::-1]
    df_utility = get_df_utility_from_df_preds(lines[0]['df'], utilities, thresholds, is_add_0_and_1=True)
    prevalence = lines[0]['df']['y'].mean() # Proportion of positive cases
    x = df_utility['threshold']
    
    # Treat All
    # Explanation of how to calculate net benefits (Page 8):
    #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2804257/pdf/nihms-108324.pdf
    # Explanation of risk thresholds / curves (Page 5): 
    #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2577036/pdf/nihms74373.pdf
    utility_perfect = prevalence * utilities['tp'] + (1 - prevalence) * utilities['tn']
    utility_all = prevalence * utilities['tp'] + (1 - prevalence) * utilities['fp']
    utility_none = prevalence * utilities['fn'] + (1 - prevalence) * utilities['tn']
    
    # Model
    y_max = np.max(df_utility['unit_utility'])
    for line in lines:
        df_preds = line['df']
        df_utility = get_df_utility_from_df_preds(df_preds, utilities, x)
        ru_none = (df_utility['unit_utility'] - utility_none) / (utility_perfect - utility_none)
        ru_all = (df_utility['unit_utility'] - utility_all) / (utility_perfect - utility_all)
        # TODO
        # y = ru_all until prevalence then ru_all
        ax.plot(x, y, label=line['label'])
        y_max = y.max() if y.max() > y_max else y_max
    
    ax.legend()
    ax.set_title(f"Relative Utility Curve", fontdict={'fontsize' : 12})
    ax.set_ylabel("Relative Utility", fontdict={'fontsize' : 10})
    ax.set_xlabel("Risk Threshold", fontdict={'fontsize' : 10})
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], y_max)
    return fig

def calc_model_performance_metrics(df_preds: pd.DataFrame,
                                   thresholds: list[float] = [0.5]) -> dict:
    """Returns a dictionary containing performance metrics for this model

    Args:
        df_preds (pd.DataFrame): Dataframe with two columns, 'y' and 'y_hat'
        thresholds (list[float]): Cutoffs for thresholding predictions to binary 0/1 (used for accuracy, Matthews correlation)
    Returns:
        dict: _description_
    """    
    pearson_corr = calc_pearsonr(df_preds['y_hat'], df_preds['y'])
    spearman_corr = calc_spearmanr(df_preds['y_hat'], df_preds['y'])
    fpr, tpr, t = sklearn.metrics.roc_curve(df_preds['y'], df_preds['y_hat'])
    auroc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, t = sklearn.metrics.precision_recall_curve(df_preds['y'], df_preds['y_hat'])
    auprc = sklearn.metrics.auc(recall, precision)
    # Threshold dependent calculations
    matthews_corr = [ sklearn.metrics.matthews_corrcoef(df_preds['y'], df_preds['y_hat'] >= x) for x in thresholds ]
    accuracy = [ np.mean(df_preds['y'] == (df_preds['y_hat'] >= x)) for x in thresholds ]
    results = {
        'auroc' : auroc,
        'auprc' : auprc,
        'pearson_corr' : pearson_corr,
        'spearman_corr' : spearman_corr,
        'spearman_corr' : spearman_corr,
        'accuracies' : accuracy,
        'matthews_corrs' : matthews_corr,
        'thresholds' : thresholds,
    }
    return results

def make_model_utility_plots(df_preds: pd.DataFrame, 
                             utilities: dict, 
                             is_show: bool = False,
                             path_to_plots_prefix: str = None):
    """Run all plotting functions
    
    Args:
        df_preds (pd.DataFrame):  - dataframe with two columns, 'y' and 'y_hat'
        is_show (bool, optional): If TRUE, show plots. Defaults to False.
        path_to_plots_prefix (str, optional): Prefix for path to saved plot files. Defaults to None.
    """
    if path_to_plots_prefix: 
        is_show = False
    if not is_show and path_to_plots_prefix is None:
        print("ERROR - Either `is_show = True` or `path_to_plots_prefix` must be specified")
        return
    plots = [
        plot_roc_curve(df_preds, utilities),
        plot_precision_recall_curve(df_preds, utilities),
        plot_ppv_v_utility_curve(df_preds, utilities),
        plot_threshold_v_utility_curve(df_preds, utilities),
        plot_work_v_utility(df_preds, utilities),
        plot_work_v_ppv_tpr_fpr(df_preds, utilities),
        plot_threshold_v_ppv_tpr_work(df_preds, utilities),
        plot_calibration_curve(df_preds),
        plot_hist_pred(df_preds),
        plot_pred_v_true(df_preds),
        plot_confusion_matrix(df_preds, utilities),
        plot_decision_curve(df_preds, utilities),
    ]
    suffixes = [
        'roc.png',
        'pr.png',
        'ppv_v_utility.png',
        'threshold_v_utility.png',
        'work_v_utility.png',
        'work_v_ppv_tpr_fpr.png',
        'threshold_v_ppv_tpr_work.png',
        'decision_curve.png',
        'calibration.png',
        'pred_histogram.png',
        'pred_v_true.png',
        'confusion_matrix.png',
    ]
    if is_show:
        plt.show() 
    else:
        for fig, suffix in zip(plots, suffixes):
            fig.savefig(path_to_plots_prefix + suffix)
        plt.close('all')

if __name__ == '__main__':
    pass