import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.interpolate import UnivariateSpline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.calibration import calibration_curve
from scipy.interpolate import UnivariateSpline
import seaborn as sns

colors = sns.color_palette("colorblind", 4)

def calculate_grouped_data(data, diagnosis_value):
    ids = data.iloc[:, 0].values
    diagnoses = data.iloc[:, 8].values
    sex = data.iloc[:, 10].values
    probabilities = data.iloc[:, -1].values
    
    group_filter = diagnoses == diagnosis_value
    filtered_data = data[group_filter]
    unique_ids = filtered_data.iloc[:, 0].unique()
    grouped_data = []
    for id_ in unique_ids:
        id_filter = filtered_data.iloc[:, 0] == id_
        patient_data = filtered_data[id_filter]
        group_probs = patient_data.iloc[:, -1].values
        avg_prob = np.mean(group_probs)
        test_count = len(group_probs)
        sex_filter = patient_data.iloc[0, 10]
        age_filter = patient_data.iloc[:, 11].max()
        DTMAXFU_filter = patient_data.iloc[:, 9].max()
        grouped_data.append([id_, diagnosis_value, avg_prob, test_count, sex_filter,
                             age_filter, DTMAXFU_filter])
    return pd.DataFrame(grouped_data, columns=['PatientID', 'is_within_5_years_of_af', 'pred', 'n', 'sex', 
                                               'Age', 'DTMAXFU'])
    
# Bootstrap function to calculate confidence intervals
def bootstrap_ci(y_true, y_pred, metric_func, num_bootstraps=1000):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    for i in range(num_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = np.percentile(sorted_scores, 2.5)
    ci_upper = np.percentile(sorted_scores, 97.5)
    return sorted_scores, ci_lower, ci_upper

# Function to calculate AUC and plot ROC and PRC curves
def plot_roc_prc(axes, df_ecg, df_patient, y_true_col_ecg, y_true_col_patient, pred_cols_ecg, pred_col_patient, titles, num_bootstraps):
    for i, pred_col in enumerate(pred_cols_ecg):
        y_true = df_ecg[y_true_col_ecg]
        y_pred = df_ecg[pred_col]        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        _, ci_lower_roc, ci_upper_roc = bootstrap_ci(y_true, y_pred, roc_auc_score, num_bootstraps)        
        axes[0].plot(fpr, tpr, lw=2, color=colors[i], label=f'{titles[i]}: {roc_auc:.2f} [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}]')        
        # PRC Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = average_precision_score(y_true, y_pred)
        _, ci_lower_prc, ci_upper_prc = bootstrap_ci(y_true, y_pred, average_precision_score, num_bootstraps)        
        axes[1].plot(recall, precision, lw=2, color=colors[i], label=f'{titles[i]}: {prc_auc:.2f} [{ci_lower_prc:.3f}, {ci_upper_prc:.3f}]')    
    # Patient-level data
    y_true_patient = df_patient[y_true_col_patient]
    y_pred_patient = df_patient[pred_col_patient]
    # ROC Curve
    fpr_patient, tpr_patient, _ = roc_curve(y_true_patient, y_pred_patient)
    roc_auc_patient = auc(fpr_patient, tpr_patient)
    _, ci_lower_roc_patient, ci_upper_roc_patient = bootstrap_ci(y_true_patient, y_pred_patient, roc_auc_score, num_bootstraps)
    axes[0].plot(fpr_patient, tpr_patient, lw=2, color=colors[len(pred_cols_ecg)], label=f'ECG-AI Patient-Level: {roc_auc_patient:.2f} [{ci_lower_roc_patient:.3f}, {ci_upper_roc_patient:.3f}]')
    # PRC Curve
    precision_patient, recall_patient, _ = precision_recall_curve(y_true_patient, y_pred_patient)
    prc_auc_patient = average_precision_score(y_true_patient, y_pred_patient)
    _, ci_lower_prc_patient, ci_upper_prc_patient = bootstrap_ci(y_true_patient, y_pred_patient, average_precision_score, num_bootstraps)
    axes[1].plot(recall_patient, precision_patient, lw=2, color=colors[len(pred_cols_ecg)], label=f'ECG-AI Patient-Level: {prc_auc_patient:.2f} [{ci_lower_prc_patient:.3f}, {ci_upper_prc_patient:.3f}]')
    # Plot settings for ROC
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=12)
    axes[0].spines['bottom'].set_color('black')
    axes[0].spines['left'].set_color('black')
    axes[0].spines['top'].set_color('black')
    axes[0].spines['right'].set_color('black')
    axes[0].tick_params(axis='x', colors='black')
    axes[0].tick_params(axis='y', colors='black')
    axes[0].grid(False)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].text(-0.1, 1.05, 'A', transform=axes[0].transAxes, fontsize=20, fontweight='bold', va='top')
    # Plot settings for PRC
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="upper right", fontsize=12)
    axes[1].spines['bottom'].set_color('black')
    axes[1].spines['left'].set_color('black')
    axes[1].spines['top'].set_color('black')
    axes[1].spines['right'].set_color('black')
    axes[1].tick_params(axis='x', colors='black')
    axes[1].tick_params(axis='y', colors='black')
    axes[1].grid(False)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].text(-0.1, 1.05, 'B', transform=axes[1].transAxes, fontsize=20, fontweight='bold', va='top')



# Function to plot calibration curve
def calculate_eci(y_true, y_pred, n_bins=10):
    # Calculate the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')    
    # Check if there are enough points to fit a spline
    if len(mean_predicted_value) < 4: 
        return np.nan    
    # Fit a smoothing spline to the calibration curve
    spline = UnivariateSpline(mean_predicted_value, fraction_of_positives, s=1)
    fitted_curve = spline(mean_predicted_value)    
    # Calculate the differences between the predicted probabilities and the fitted calibration curve
    differences = mean_predicted_value - fitted_curve    
    # Compute the ECI
    eci = np.sqrt(np.mean(differences ** 2))
    return eci

def bootstrap_eci(y_true, y_pred, num_bootstraps=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    eci_values = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(num_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        eci = calculate_eci(y_true[indices], y_pred[indices])
        if not np.isnan(eci):
            eci_values.append(eci)
    sorted_eci = np.array(eci_values)
    sorted_eci.sort()
    ci_lower = np.percentile(sorted_eci, 2.5)
    ci_upper = np.percentile(sorted_eci, 97.5)
    return np.mean(sorted_eci), ci_lower, ci_upper

def modified_calibration_curve(ax, df, df_patient, set_value, pred_cols, titles, num_bootstraps=1000):
    colors = sns.color_palette("colorblind", len(pred_cols) + 1)
    for i, pred_col in enumerate(pred_cols):
        # Filter the data
        subset = df
        y_val = subset['is_within_5_years_of_af'].values
        y_val_pred = subset[pred_col].values
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_pred, n_bins=10, strategy='uniform')
        # Fit a smoothing spline to the calibration curve
        if len(mean_predicted_value) >= 4: 
            spline = UnivariateSpline(mean_predicted_value, fraction_of_positives, s=1)
            spline_fitted_curve = spline(mean_predicted_value)
            # Calculate ECI using bootstrapping
            eci_mean, eci_lower, eci_upper = bootstrap_eci(y_val, y_val_pred, num_bootstraps)
            # Plot spline smoothed calibration curve
            ax.plot(mean_predicted_value, spline_fitted_curve, color=colors[i], label=f'{titles[i]}: ECI = {eci_mean:.3f} [{eci_lower:.3f}, {eci_upper:.3f}]')
    # Patient-level data
    subset_patient = df_patient
    y_val_patient = subset_patient['is_within_5_years_of_af'].values
    y_val_pred_patient = subset_patient['pred'].values
    # Calculate calibration curve
    fraction_of_positives_patient, mean_predicted_value_patient = calibration_curve(y_val_patient, y_val_pred_patient, n_bins=10, strategy='uniform')
    # Fit a smoothing spline to the calibration curve
    if len(mean_predicted_value_patient) >= 4: 
        spline_patient = UnivariateSpline(mean_predicted_value_patient, fraction_of_positives_patient, s=1)
        spline_fitted_curve_patient = spline_patient(mean_predicted_value_patient)
        # Calculate ECI using bootstrapping
        eci_mean_patient, eci_lower_patient, eci_upper_patient = bootstrap_eci(y_val_patient, y_val_pred_patient, num_bootstraps)
        # Plot spline smoothed calibration curve for patient-level data
        ax.plot(mean_predicted_value_patient, spline_fitted_curve_patient, color=colors[-1], label=f'ECG-AI Patient-Level: ECI = {eci_mean_patient:.3f} [{eci_lower_patient:.3f}, {eci_upper_patient:.3f}]')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_ylabel('Observed 5-year Risk', fontsize=12)
    ax.set_xlabel('Predicted 5-year Risk', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')


# Function to calculate net benefit
def calculate_net_benefit(y_true, y_pred_prob, thresholds):
    net_benefits = []
    for threshold in thresholds:
        predicted_positive = y_pred_prob >= threshold
        true_positive = (y_true == 1) & (predicted_positive == 1)
        false_positive = (y_true == 0) & (predicted_positive == 1)        
        tp = np.sum(true_positive)
        fp = np.sum(false_positive)
        tn = np.sum((y_true == 0) & (predicted_positive == 0))
        fn = np.sum((y_true == 1) & (predicted_positive == 0))        
        n = len(y_true)
        nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(nb)    
    return np.array(net_benefits)

# Function to plot Decision Curve Analysis (DCA)
def plot_dca(ax, df, df_patient, set_value, pred_cols, titles):
    sns.set(style="whitegrid")
    # Filter the data for the specified set
    subset = df
    y_true = subset['is_within_5_years_of_af'].values
    colors = sns.color_palette("colorblind", len(pred_cols) + 3)
    # Plot net benefit curves for each model
    for i, pred_col in enumerate(pred_cols):
        y_pred_prob = subset[pred_col].values
        net_benefit = calculate_net_benefit(y_true, y_pred_prob, thresholds=np.arange(0.01, 1.0, 0.01))
        ax.plot(np.arange(0.01, 1.0, 0.01), net_benefit, color=colors[i], label=f'ECG {titles[i]}')
    # Patient-level data
    subset_patient = df_patient
    y_true_patient = subset_patient['is_within_5_years_of_af'].values
    y_pred_prob_patient = subset_patient['pred'].values
    # Calculate net benefit for patient-level data
    net_benefit_patient = calculate_net_benefit(y_true_patient, y_pred_prob_patient, thresholds=np.arange(0.01, 1.0, 0.01))
    ax.plot(np.arange(0.01, 1.0, 0.01), net_benefit_patient, color=colors[len(pred_cols)], label='ECG-AI Patient-Level')
    # Plot treat all and treat none lines
    treat_all_net_benefit = np.array([(np.sum(y_true) / len(y_true)) - (t / (1 - t)) * (np.sum(y_true == 0) / len(y_true)) for t in np.arange(0.01, 1.0, 0.01)])
    treat_all_net_benefit_patient = np.array([(np.sum(y_true_patient) / len(y_true_patient)) - (t / (1 - t)) * (np.sum(y_true_patient == 0) / len(y_true_patient)) for t in np.arange(0.01, 1.0, 0.01)])
    treat_none_net_benefit = np.zeros(len(np.arange(0.01, 1.0, 0.01)))
    ax.plot(np.arange(0.01, 1.0, 0.01), treat_all_net_benefit, color='k', linestyle='--', label='Screen All')
    ax.plot(np.arange(0.01, 1.0, 0.01), treat_none_net_benefit, color='k', linestyle='-.', label='Screen None')
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_xlim([-0.005, 0.4])
    ax.set_ylim([-0.005, 0.16])
    ax.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(False)
    # Black border
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    # Add the prevalence number to the plot at the y axis position of 0.01 and x axis position of prevalence
    prevalence_ecg = np.sum(y_true) / len(y_true)
    prevalence_patient = np.sum(y_true_patient) / len(y_true_patient)
    ax.text(prevalence_ecg, 0.0, f'{prevalence_ecg:.2f}', fontsize=12, color='black', ha='left', va='bottom')
    ax.text(-0.1, 1.05, 'D', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

def plot_all(df, df_patient, set_value, pred_cols, titles, suptitle, num_bootstraps=1000, output_path='plot.png'):
    sns.set(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})    
    plot_roc_prc(axes[0], df, df_patient, 'is_within_5_years_of_af', 'is_within_5_years_of_af', pred_cols, 'pred', titles, num_bootstraps)
    modified_calibration_curve(axes[1][0], df, df_patient, set_value, pred_cols, titles, num_bootstraps)
    plot_dca(axes[1][1], df, df_patient, set_value, pred_cols, titles)    
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

def calculate_hazard_ratio(high_risk, low_risk):
    combined = pd.concat([high_risk, low_risk])
    combined['risk_group'] = np.where(combined['pred'] >= 0.12, 1, 0)  # Encode 'High Risk' as 1 and 'Low Risk' as 0
    combined['duration'] = combined['duration'].astype(float)
    combined['event'] = combined['event'].astype(bool)
    cph = CoxPHFitter()
    cph.fit(combined[['duration', 'event', 'risk_group']], duration_col='duration', event_col='event')    
    hr = cph.hazard_ratios_['risk_group']
    ci_lower = np.exp(cph.confidence_intervals_.loc['risk_group', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['risk_group', '95% upper-bound'])
    print("Hazard Ratio:", hr)
    print("95% Confidence Interval Lower Bound:", ci_lower)
    print("95% Confidence Interval Upper Bound:", ci_upper)   
    return hr, ci_lower, ci_upper

def plot_kaplan_meier(ax, high_risk, low_risk, title, label, show_legend=False):
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()    
    # Fit the data
    kmf_high.fit(high_risk['duration'] / 365.25, event_observed=high_risk['event'], label='ECG-AI High Risk')
    kmf_low.fit(low_risk['duration'] / 365.25, event_observed=low_risk['event'], label='ECG-AI Low Risk')    
    # Plot the survival functions
    kmf_high.plot_survival_function(ax=ax, ci_show=True, linestyle='-', color=colors[0])
    kmf_low.plot_survival_function(ax=ax, ci_show=True, linestyle='-', color=colors[3])        
    # Calculate the log-rank test p-value
    results = logrank_test(high_risk['duration'], low_risk['duration'], event_observed_A=high_risk['event'], event_observed_B=low_risk['event'])
    p_value = results.p_value
    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"        
    # Calculate the hazard ratio
    hr, ci_lower, ci_upper = calculate_hazard_ratio(high_risk, low_risk)
    hr_text = f"HR={hr:.2f} (95% CI {ci_lower:.2f}-{ci_upper:.2f})"        
    ax.set_xlabel('Years')
    ax.set_ylabel('Incident-free Probability')
    ax.grid(False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0., 1)
    ax.set_xticks(np.arange(0, 11, 2))  # Ensure the xticks go from 0 to 10
    ax.set_xticklabels(np.arange(0, 11, 2))  # Ensure the xtick labels go from 0 to 10
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')
    ax.text(0.5, 0.07, f"{hr_text}", transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.02, f"{p_text}", transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='center', bbox=dict(facecolor='white', alpha=0.5))        
    if show_legend:
        ax.legend()
    else:
        ax.get_legend().remove()    

def add_number_at_risk(ax, kmf_high, kmf_low, label_positions):
    times = np.arange(0, 11, 2)
    high_risk_counts = []
    low_risk_counts = []
    for time in label_positions:
        closest_high = kmf_high.event_table.index[kmf_high.event_table.index.searchsorted(time, side='right') - 1]
        closest_low = kmf_low.event_table.index[kmf_low.event_table.index.searchsorted(time, side='right') - 1]
        high_risk_counts.append(kmf_high.event_table.at[closest_high, 'at_risk'])
        low_risk_counts.append(kmf_low.event_table.at[closest_low, 'at_risk'])
    ax.text(0, -0.15, "At risk", ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.transAxes)
    for i, (x, high, low) in enumerate(zip(times, high_risk_counts, low_risk_counts)):
        ax.text(x / 10, -0.25, f'{int(high)}', ha='center', va='top', fontsize=12, color=colors[0], transform=ax.transAxes)
        ax.text(x / 10, -0.20, f'{int(low)}', ha='center', va='top', fontsize=12, color=colors[3], transform=ax.transAxes)
