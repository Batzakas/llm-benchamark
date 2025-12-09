import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import os
import pandas as pd

def check_assumptions(model_fit, metric_name: str, block_name: str, output_dir: str = "diagnostics"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    residuals = model_fit.resid
    fitted_values = model_fit.fittedvalues

    # 1. Normality Check (Shapiro-Wilk)
    # H0: Data is normal. p < 0.05 means NOT normal.
    shapiro_stat, shapiro_p = shapiro(residuals)
    
    # 2. Homoscedasticity Check (Breusch-Pagan)
    # H0: Variance is constant. p < 0.05 means Heteroscedastic (bad).
    # BP requires the design matrix (exog)
    bp_test = het_breuschpagan(residuals, model_fit.model.exog)
    bp_p = bp_test[1]

    print(f"    [Diagnostics] {metric_name}:")
    print(f"      - Normality (Shapiro): p={shapiro_p:.4f} {'(Warn)' if shapiro_p < 0.05 else '(OK)'}")
    print(f"      - Equal Var (Breusch): p={bp_p:.4f} {'(Warn)' if bp_p < 0.05 else '(OK)'}")

    # 3. Diagnostic Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Q Plot (Normality)
    sm.qqplot(residuals, line='s', ax=ax[0])
    ax[0].set_title(f"Q-Q Plot: {metric_name} ({block_name})")

    # Residuals vs Fitted (Homoscedasticity)
    ax[1].scatter(fitted_values, residuals, alpha=0.5)
    ax[1].axhline(0, color='red', linestyle='--')
    ax[1].set_xlabel("Fitted Values")
    ax[1].set_ylabel("Residuals")
    ax[1].set_title(f"Residuals vs Fitted: {metric_name}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{block_name}_{metric_name}_diag.png")
    plt.close()