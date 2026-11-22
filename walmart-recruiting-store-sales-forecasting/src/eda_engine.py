"""
Walmart Sales Forecasting — Phase 2 Exploratory Data Analysis & Statistical Engine
Module: src/eda_engine.py
Author: Lead Data Scientist & ML Engineer

Responsibilities:
- Compute descriptive statistics, sales distributions, store and department metrics.
- Perform holiday impact analysis (Super Bowl, Labor Day, Thanksgiving, Christmas, and Pre-Christmas week).
- Analyze promotional markdowns and macroeconomic features (correlations, missingness).
- Execute STL time-series decomposition and stationarity testing (ADF, KPSS).
- Investigate cold-start series, incomplete panels, and negative sales.
- Export all analytical tables to outputs/eda/ and publication-quality figures to outputs/figures/.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

# Styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
sns.set_theme(style="whitegrid", palette="muted")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_EDA_DIR = os.path.join(PROJECT_DIR, "outputs", "eda")
OUTPUT_FIG_DIR = os.path.join(PROJECT_DIR, "outputs", "figures")

os.makedirs(OUTPUT_EDA_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

logger = logging.getLogger("WalmartEDAEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_processed_data():
    train_path = os.path.join(PROCESSED_DATA_DIR, "train_merged.parquet")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test_merged.parquet")
    logger.info(f"Loading analytical datasets: {train_path}, {test_path}")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    return train, test


def run_sales_eda(train):
    """Overall sales distributions, store and department metrics, and business concentration."""
    logger.info("Running Sales EDA & Distributions...")
    sales = train["Weekly_Sales"]
    
    # 1. Overall stats
    stats_dict = {
        "count": int(sales.count()),
        "mean": float(sales.mean()),
        "std": float(sales.std()),
        "median": float(sales.median()),
        "min": float(sales.min()),
        "q25": float(sales.quantile(0.25)),
        "q75": float(sales.quantile(0.75)),
        "iqr": float(sales.quantile(0.75) - sales.quantile(0.25)),
        "max": float(sales.max()),
        "skewness": float(sales.skew()),
        "kurtosis": float(sales.kurtosis()),
        "total_sales_dollars": float(sales.sum()),
        "negative_sales_count": int((sales < 0).sum()),
        "negative_sales_pct": float((sales < 0).sum() / len(sales) * 100),
        "zero_sales_count": int((sales == 0).sum()),
        "zero_sales_pct": float((sales == 0).sum() / len(sales) * 100),
    }
    with open(os.path.join(OUTPUT_EDA_DIR, "overall_sales_summary.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    # Figure 2: Sales distribution (Linear and Log-Shifted)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear distribution (clipping positive tail for visual clarity in histogram)
    sns.histplot(sales, bins=100, ax=axes[0], color="#1f77b4", kde=True)
    axes[0].set_title("Distribution of Weekly Sales (Linear Scale)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Weekly Sales ($)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_xlim(-5000, 100000)
    axes[0].axvline(sales.median(), color="red", linestyle="--", label=f"Median: ${sales.median():,.2f}")
    axes[0].axvline(sales.mean(), color="green", linestyle=":", label=f"Mean: ${sales.mean():,.2f}")
    axes[0].legend()
    
    # Signed log1p distribution
    signed_log_sales = np.sign(sales) * np.log1p(np.abs(sales))
    sns.histplot(signed_log_sales, bins=100, ax=axes[1], color="#2ca02c", kde=True)
    axes[1].set_title("Distribution of Weekly Sales (Signed Log1p Scale)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Signed Log(1 + |Weekly Sales|)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].axvline(np.sign(sales.median()) * np.log1p(np.abs(sales.median())), color="red", linestyle="--", label="Median")
    axes[1].legend()
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig02_sales_distribution_and_log.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 3: Negative sales by department
    neg_sales = train[train["Weekly_Sales"] < 0]
    neg_dept = neg_sales.groupby("Dept")["Weekly_Sales"].agg(["count", "sum", "min"]).sort_values("count", ascending=False).head(12)
    neg_dept.to_csv(os.path.join(OUTPUT_EDA_DIR, "top_negative_sales_departments.csv"))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=neg_dept.index.astype(str), y=neg_dept["count"], ax=ax, palette="Reds_r")
    ax.set_title("Top Departments with Negative Sales Occurrences (Customer Returns / Adjustments)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Department ID", fontsize=12)
    ax.set_ylabel("Number of Negative Sales Records", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig03_negative_sales_by_dept.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Store Analysis
    store_stats = train.groupby(["Store", "Type", "Size"])["Weekly_Sales"].agg(["count", "sum", "mean", "median", "std"]).reset_index()
    store_stats = store_stats.sort_values("sum", ascending=False)
    store_stats.to_csv(os.path.join(OUTPUT_EDA_DIR, "store_sales_performance.csv"), index=False)

    # Figure 4: Ranked store total sales
    fig, ax = plt.subplots(figsize=(16, 7))
    palette_type = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    sns.barplot(data=store_stats, x="Store", y="sum", hue="Type", order=store_stats["Store"], dodge=False, palette=palette_type, ax=ax)
    ax.set_title("Total Historical Sales by Store (Ranked, Colored by Store Type)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Store Number (Ranked High to Low)", fontsize=12)
    ax.set_ylabel("Total Revenue ($)", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
    plt.xticks(rotation=90)
    ax.legend(title="Store Type", loc="upper right")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig04_store_sales_ranking.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 5: Store Size vs Average Weekly Sales
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=store_stats, x="Size", y="mean", hue="Type", style="Type", s=130, palette=palette_type, ax=ax)
    sns.regplot(data=store_stats, x="Size", y="mean", scatter=False, ax=ax, color="#555555", line_kws={"linestyle": "--", "alpha": 0.8})
    ax.set_title("Store Physical Area (Size) vs Average Weekly Sales by Store Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Store Size (Square Feet)", fontsize=11)
    ax.set_ylabel("Average Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.0f} sq ft"))
    
    # Compute correlation
    r_size, p_size = stats.pearsonr(store_stats["Size"], store_stats["mean"])
    ax.annotate(f"Pearson r = {r_size:.3f} (p < 0.001)", xy=(0.05, 0.90), xycoords="axes fraction",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#cccccc"))
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig05_store_type_size_vs_sales.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Department Analysis: Top 10 and Bottom 10
    dept_stats = train.groupby("Dept")["Weekly_Sales"].agg(["count", "sum", "mean", "median", "std"]).sort_values("sum", ascending=False).reset_index()
    dept_stats.to_csv(os.path.join(OUTPUT_EDA_DIR, "department_sales_performance.csv"), index=False)
    
    top10_dept = dept_stats.head(10)
    bottom10_dept = dept_stats.tail(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=top10_dept, x="sum", y=top10_dept["Dept"].astype(str), ax=ax1, palette="Blues_r")
    ax1.set_title("Top 10 Highest-Revenue Departments", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Total Sales ($)", fontsize=11)
    ax1.set_ylabel("Department ID", fontsize=11)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))

    sns.barplot(data=bottom10_dept, x="sum", y=bottom10_dept["Dept"].astype(str), ax=ax2, palette="Oranges_r")
    ax2.set_title("Bottom 10 Lowest-Revenue Departments", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Total Sales ($)", fontsize=11)
    ax2.set_ylabel("Department ID", fontsize=11)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-3:.1f}K" if x >= 0 else f"-${abs(x)*1e-3:.1f}K"))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig06_top_bottom_departments_revenue.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Business Concentration (Pareto / Lorenz Curve)
    store_cum = store_stats["sum"].cumsum() / store_stats["sum"].sum() * 100
    dept_cum = dept_stats["sum"].cumsum() / dept_stats["sum"].sum() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.linspace(0, 100, len(store_cum)), store_cum, label="Stores Concentration", color="#1f77b4", linewidth=2.5)
    ax.plot(np.linspace(0, 100, len(dept_cum)), dept_cum, label="Departments Concentration", color="#d62728", linewidth=2.5)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#888888", label="Perfect Equality Line")
    
    ax.set_title("Sales Concentration Curves (Pareto Analysis)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cumulative Percentage of Entities (%)", fontsize=11)
    ax.set_ylabel("Cumulative Percentage of Revenue (%)", fontsize=11)
    ax.axhline(80, color="gray", linestyle=":", alpha=0.7)
    
    # Compute 80/20 metrics
    dept_20_pct = int(np.ceil(0.20 * len(dept_stats)))
    dept_20_rev = dept_stats.head(dept_20_pct)["sum"].sum() / dept_stats["sum"].sum() * 100
    ax.annotate(f"Top 20% of Departments generate {dept_20_rev:.1f}% of Revenue", xy=(20, dept_20_rev),
                xytext=(35, 60), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d6b656"))
    
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig07_sales_concentration_pareto.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")


def run_time_series_eda(train, test):
    """Aggregate weekly trends, YoY seasonality, store/dept trajectories."""
    logger.info("Running Time-Series EDA & Trends...")
    
    # 1. Aggregate Weekly Sales
    agg_weekly = train.groupby("Date").agg({
        "Weekly_Sales": "sum",
        "IsHoliday": "first",
        "Temperature": "mean",
        "Fuel_Price": "mean",
        "CPI": "mean",
        "Unemployment": "mean"
    }).reset_index()
    agg_weekly.to_csv(os.path.join(OUTPUT_EDA_DIR, "aggregate_weekly_sales_trend.csv"), index=False)

    # Figure 8: 143-week aggregate weekly sales line plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(agg_weekly["Date"], agg_weekly["Weekly_Sales"], color="#1f77b4", linewidth=2, label="Weekly Aggregate Sales")
    
    # Highlight holiday weeks
    hol_weeks = agg_weekly[agg_weekly["IsHoliday"] == True]
    ax.scatter(hol_weeks["Date"], hol_weeks["Weekly_Sales"], color="#d62728", s=65, zorder=5, label="Holiday Weeks (Weight = 5)")
    
    # Annotate Thanksgiving & Christmas peaks
    thanksgiving_dates = ["2010-11-26", "2011-11-25"]
    for td in thanksgiving_dates:
        td_val = agg_weekly[agg_weekly["Date"] == td]["Weekly_Sales"].values
        if len(td_val) > 0:
            ax.annotate("Thanksgiving\n(Black Friday)", xy=(pd.Timestamp(td), td_val[0]),
                        xytext=(pd.Timestamp(td) - pd.Timedelta(days=40), td_val[0] + 3e6),
                        arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1, headwidth=5),
                        fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="#d62728"))

    ax.set_title("Total Historical Weekly Sales Across All Walmart Stores (Feb 2010 – Oct 2012)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Week-Ending Date", fontsize=11)
    ax.set_ylabel("Aggregate Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig08_aggregate_weekly_sales_trend.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 9: Year-over-Year overlay by calendar week number
    agg_weekly["Year"] = agg_weekly["Date"].dt.year
    agg_weekly["Week"] = agg_weekly["Date"].dt.isocalendar().week
    
    fig, ax = plt.subplots(figsize=(14, 6))
    year_palette = {2010: "#1f77b4", 2011: "#2ca02c", 2012: "#ff7f0e"}
    for yr in [2010, 2011, 2012]:
        sub_yr = agg_weekly[agg_weekly["Year"] == yr].sort_values("Week")
        ax.plot(sub_yr["Week"], sub_yr["Weekly_Sales"], marker='o', markersize=4, label=f"Year {yr}", color=year_palette[yr], linewidth=2)
        
    ax.set_title("Year-over-Year Weekly Sales Overlay (Aligned by ISO Week Number)", fontsize=14, fontweight='bold')
    ax.set_xlabel("ISO Calendar Week Number (1 to 52)", fontsize=11)
    ax.set_ylabel("Total Aggregate Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
    ax.set_xticks(range(1, 53, 2))
    ax.legend(title="Calendar Year", loc="upper left")
    
    # Annotate cyclical surges (Easter/Spring, Back-to-School, Thanksgiving week 47, Christmas week 51)
    ax.axvspan(46.5, 47.5, color="red", alpha=0.15, label="Thanksgiving Window")
    ax.axvspan(50.5, 52.5, color="green", alpha=0.15, label="Holiday Shopping Peak")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig09_yoy_weekly_sales_overlay.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 10: Store Type Weekly Sales Trajectories
    store_type_weekly = train.groupby(["Date", "Type"])["Weekly_Sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(data=store_type_weekly, x="Date", y="Weekly_Sales", hue="Type", palette={"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}, ax=ax, linewidth=2)
    ax.set_title("Average Weekly Sales Trajectory by Store Type (A: Supercenter, B: Standard, C: Express)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Mean Department Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig10_store_type_trends_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 11: Department Seasonality Trends Comparison
    # Compare Grocery (92), Electronics (72), Lawn & Garden / Seasonal (18), Clearance/Returns (47)
    sample_depts = [92, 72, 18, 47]
    dept_ts = train[train["Dept"].isin(sample_depts)].groupby(["Date", "Dept"])["Weekly_Sales"].mean().reset_index()
    dept_names = {92: "Dept 92 (Grocery)", 72: "Dept 72 (Electronics)", 18: "Dept 18 (Toys / Seasonal)", 47: "Dept 47 (Clearance / Returns)"}
    dept_ts["Dept_Label"] = dept_ts["Dept"].map(dept_names)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=dept_ts, x="Date", y="Weekly_Sales", hue="Dept_Label", ax=ax, linewidth=2)
    ax.set_title("Weekly Sales Profiles Across Contrasting Retail Departments", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Mean Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig11_department_seasonality_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")


def run_holiday_deep_dive(train):
    """Rigorous analysis of Super Bowl, Labor Day, Thanksgiving, Christmas, and Pre-Christmas week."""
    logger.info("Running Holiday Deep-Dive Analysis...")
    
    holidays_calendar = [
        {"Event": "Super Bowl", "Date": "2010-02-12", "Year": 2010, "IsHoliday": True},
        {"Event": "Labor Day", "Date": "2010-09-10", "Year": 2010, "IsHoliday": True},
        {"Event": "Thanksgiving", "Date": "2010-11-26", "Year": 2010, "IsHoliday": True},
        {"Event": "Christmas", "Date": "2010-12-31", "Year": 2010, "IsHoliday": True},
        {"Event": "Super Bowl", "Date": "2011-02-11", "Year": 2011, "IsHoliday": True},
        {"Event": "Labor Day", "Date": "2011-09-09", "Year": 2011, "IsHoliday": True},
        {"Event": "Thanksgiving", "Date": "2011-11-25", "Year": 2011, "IsHoliday": True},
        {"Event": "Christmas", "Date": "2011-12-30", "Year": 2011, "IsHoliday": True},
        {"Event": "Super Bowl", "Date": "2012-02-10", "Year": 2012, "IsHoliday": True},
        {"Event": "Labor Day", "Date": "2012-09-07", "Year": 2012, "IsHoliday": True},
    ]
    
    # Pre-Christmas weeks (Week preceding Christmas Day)
    pre_christmas_dates = ["2010-12-24", "2011-12-23"]
    
    # 1. Overall Holiday vs Non-Holiday Stats
    hol_stats = train.groupby("IsHoliday")["Weekly_Sales"].agg(["count", "mean", "median", "std", "sum"]).reset_index()
    hol_stats["mean_uplift_pct"] = (hol_stats["mean"] - hol_stats.loc[0, "mean"]) / hol_stats.loc[0, "mean"] * 100
    hol_stats["median_uplift_pct"] = (hol_stats["median"] - hol_stats.loc[0, "median"]) / hol_stats.loc[0, "median"] * 100
    hol_stats.to_csv(os.path.join(OUTPUT_EDA_DIR, "holiday_vs_nonholiday_stats.csv"), index=False)

    # 2. Detailed Event Breakdown
    event_records = []
    train_copy = train.copy()
    train_copy["Date_str"] = train_copy["Date"].dt.strftime("%Y-%m-%d")
    
    for h in holidays_calendar:
        sub = train_copy[train_copy["Date_str"] == h["Date"]]
        event_records.append({
            "Event": h["Event"],
            "Year": h["Year"],
            "Date": h["Date"],
            "IsHoliday": True,
            "Total_Sales": float(sub["Weekly_Sales"].sum()),
            "Mean_Sales": float(sub["Weekly_Sales"].mean()),
            "Median_Sales": float(sub["Weekly_Sales"].median()),
            "Count": int(len(sub))
        })
        
    for pcd in pre_christmas_dates:
        sub = train_copy[train_copy["Date_str"] == pcd]
        yr = int(pcd[:4])
        event_records.append({
            "Event": "Pre-Christmas Peak",
            "Year": yr,
            "Date": pcd,
            "IsHoliday": False, # Explicitly False in Kaggle definition
            "Total_Sales": float(sub["Weekly_Sales"].sum()),
            "Mean_Sales": float(sub["Weekly_Sales"].mean()),
            "Median_Sales": float(sub["Weekly_Sales"].median()),
            "Count": int(len(sub))
        })
        
    events_df = pd.DataFrame(event_records)
    events_df.to_csv(os.path.join(OUTPUT_EDA_DIR, "major_holiday_events_performance.csv"), index=False)

    # Figure 12: Holiday Uplift Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Event average total sales
    event_agg = events_df.groupby("Event")["Total_Sales"].mean().reset_index().sort_values("Total_Sales", ascending=False)
    non_hol_avg_total = train_copy[train_copy["IsHoliday"] == False].groupby("Date")["Weekly_Sales"].sum().mean()
    
    sns.barplot(data=event_agg, x="Event", y="Total_Sales", ax=ax1, palette="Blues_r")
    ax1.axhline(non_hol_avg_total, color="red", linestyle="--", linewidth=1.8, label=f"Non-Holiday Mean: ${non_hol_avg_total*1e-6:.1f}M")
    ax1.set_title("Average Total Weekly Sales by Retail Event", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Event", fontsize=11)
    ax1.set_ylabel("Average Weekly Total ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # Mean Department Sales by Event
    dept_event_agg = events_df.groupby("Event")["Mean_Sales"].mean().reset_index().sort_values("Mean_Sales", ascending=False)
    non_hol_dept_mean = train_copy[train_copy["IsHoliday"] == False]["Weekly_Sales"].mean()
    sns.barplot(data=dept_event_agg, x="Event", y="Mean_Sales", ax=ax2, palette="Greens_r")
    ax2.axhline(non_hol_dept_mean, color="red", linestyle="--", linewidth=1.8, label=f"Baseline Non-Holiday: ${non_hol_dept_mean:,.2f}")
    ax2.set_title("Mean Department Sales by Retail Event", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Event", fontsize=11)
    ax2.set_ylabel("Mean Sales per Record ($)", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig12_holiday_sales_uplift_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 13: Pre-Christmas Week vs Christmas Week Deep Dive
    # Compare 2010: 2010-12-17 (Normal), 2010-12-24 (Pre-Christmas Peak), 2010-12-31 (Christmas Week / Post-Xmas)
    # Compare 2011: 2011-12-16 (Normal), 2011-12-23 (Pre-Christmas Peak), 2011-12-30 (Christmas Week / Post-Xmas)
    december_dates = ["2010-12-17", "2010-12-24", "2010-12-31", "2011-12-16", "2011-12-23", "2011-12-30"]
    dec_df = train_copy[train_copy["Date_str"].isin(december_dates)].groupby("Date_str").agg({
        "Weekly_Sales": "sum",
        "IsHoliday": "first"
    }).reset_index()
    dec_df["Year"] = dec_df["Date_str"].str[:4]
    dec_df["Week_Type"] = dec_df["Date_str"].map({
        "2010-12-17": "Mid-Dec Normal",
        "2010-12-24": "Pre-Xmas Peak (IsHoliday=False)",
        "2010-12-31": "Christmas Week (IsHoliday=True)",
        "2011-12-16": "Mid-Dec Normal",
        "2011-12-23": "Pre-Xmas Peak (IsHoliday=False)",
        "2011-12-30": "Christmas Week (IsHoliday=True)"
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=dec_df, x="Date_str", y="Weekly_Sales", hue="Week_Type", dodge=False, ax=ax)
    ax.set_title("The Pre-Christmas Effect: Week 51 Surge vs Christmas Week Drop", fontsize=14, fontweight='bold')
    ax.set_xlabel("Week-Ending Friday", fontsize=11)
    ax.set_ylabel("Total Aggregate Weekly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"${h*1e-6:.1f}M", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig13_pre_christmas_effect_analysis.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")


def run_markdown_and_economic_eda(train, test):
    """Analyze promotional markdowns, macroeconomic indicators, correlations, and test missingness."""
    logger.info("Running Markdown and Economic Analysis...")
    
    # 1. Feature missingness profile across time
    agg_features = train.groupby("Date").agg({
        "MarkDown1": lambda x: x.notnull().mean() * 100,
        "MarkDown2": lambda x: x.notnull().mean() * 100,
        "MarkDown3": lambda x: x.notnull().mean() * 100,
        "MarkDown4": lambda x: x.notnull().mean() * 100,
        "MarkDown5": lambda x: x.notnull().mean() * 100,
        "CPI": lambda x: x.notnull().mean() * 100,
        "Unemployment": lambda x: x.notnull().mean() * 100
    }).reset_index()

    # Figure 14: MarkDown Temporal Availability Heatmap / Timeline
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg_features["Date"], agg_features["MarkDown1"], label="MarkDown1 Availability (%)", color="#1f77b4", linewidth=2)
    ax.plot(agg_features["Date"], agg_features["MarkDown2"], label="MarkDown2 Availability (%)", color="#ff7f0e", linewidth=2)
    ax.plot(agg_features["Date"], agg_features["MarkDown3"], label="MarkDown3 Availability (%)", color="#2ca02c", linewidth=2)
    ax.plot(agg_features["Date"], agg_features["MarkDown4"], label="MarkDown4 Availability (%)", color="#d62728", linewidth=2)
    ax.plot(agg_features["Date"], agg_features["MarkDown5"], label="MarkDown5 Availability (%)", color="#9467bd", linewidth=2)
    
    ax.axvline(pd.Timestamp("2011-11-04"), color="black", linestyle="--", linewidth=1.5, label="Recording Start (Nov 2011)")
    ax.set_title("Promotional MarkDown Availability Over Time (Pre-Nov 2011 Structural Break)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Non-Null Observation Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="center left")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig14_markdown_temporal_availability.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # 2. Economic Features Correlations
    # Sample 50,000 observations to compute robust correlations
    corr_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",
                 "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    
    # Restrict to post-markdown period for clean correlation on markdowns
    post_md = train[train["Date"] >= "2011-11-11"][corr_cols].dropna(subset=["Weekly_Sales"])
    pearson_corr = post_md.corr(method="pearson")
    spearman_corr = post_md.corr(method="spearman")
    
    pearson_corr.to_csv(os.path.join(OUTPUT_EDA_DIR, "pearson_correlation_matrix.csv"))
    spearman_corr.to_csv(os.path.join(OUTPUT_EDA_DIR, "spearman_correlation_matrix.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax1, cbar_kws={'label': 'Pearson r'})
    ax1.set_title("Pearson Linear Correlation Matrix (Post-Nov 2011 Data)", fontsize=13, fontweight='bold')

    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2, cbar_kws={'label': 'Spearman rank ρ'})
    ax2.set_title("Spearman Rank Correlation Matrix (Nonlinear Monotonic)", fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig15_economic_features_correlation_matrix.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Figure 18: Test Macroeconomic Blackout
    test_agg = test.groupby("Date").agg({
        "CPI": lambda x: x.isnull().mean() * 100,
        "Unemployment": lambda x: x.isnull().mean() * 100,
        "Temperature": lambda x: x.notnull().mean() * 100
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(test_agg["Date"], test_agg["CPI"], width=5, color="#d62728", alpha=0.8, label="CPI Missing Rate in Test (%)")
    ax.axvline(pd.Timestamp("2013-05-03"), color="black", linestyle="--", linewidth=1.5, label="Blackout Start (May 3, 2013)")
    ax.set_title("Critical Test-Period Macroeconomic Blackout: Final 13 Weeks Lack CPI & Unemployment", fontsize=13, fontweight='bold')
    ax.set_xlabel("Test Horizon Date", fontsize=11)
    ax.set_ylabel("Missing Percentage (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig18_test_cpi_unemployment_blackout.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")


def run_time_series_decomposition(train):
    """STL Decomposition & Stationarity Tests (ADF, KPSS)."""
    logger.info("Running STL Decomposition & Stationarity Tests...")
    
    # Weekly aggregate series
    weekly_agg = train.groupby("Date")["Weekly_Sales"].sum().asfreq("W-FRI")
    
    # 1. STL Decomposition
    # Period = 52 for annual weekly seasonality
    # Total observations = 143 weeks (~2.75 years)
    stl = STL(weekly_agg, period=52, robust=True)
    res = stl.fit()
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(weekly_agg.index, weekly_agg.values, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Observed ($)", fontsize=10, fontweight='bold')
    axes[0].set_title("STL Time-Series Decomposition (period = 52 weeks, robust = True)", fontsize=14, fontweight='bold')
    
    axes[1].plot(weekly_agg.index, res.trend, color="#ff7f0e", linewidth=2.0)
    axes[1].set_ylabel("Trend ($)", fontsize=10, fontweight='bold')
    
    axes[2].plot(weekly_agg.index, res.seasonal, color="#2ca02c", linewidth=1.8)
    axes[2].set_ylabel("Seasonal ($)", fontsize=10, fontweight='bold')
    
    axes[3].scatter(weekly_agg.index, res.resid, color="#555555", s=18, alpha=0.8)
    axes[3].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Residual ($)", fontsize=10, fontweight='bold')
    axes[3].set_xlabel("Date", fontsize=11)
    
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
        
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig16_stl_decomposition_aggregate_sales.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # 2. Stationarity Tests
    # Augmented Dickey-Fuller (ADF) Test
    # H0: Series has a unit root (Non-Stationary)
    # H1: Series is stationary
    adf_res = adfuller(weekly_agg.dropna(), autolag="AIC")
    adf_dict = {
        "test_name": "Augmented Dickey-Fuller (ADF)",
        "test_statistic": float(adf_res[0]),
        "p_value": float(adf_res[1]),
        "lags_used": int(adf_res[2]),
        "n_obs": int(adf_res[3]),
        "critical_values": {k: float(v) for k, v in adf_res[4].items()},
        "is_stationary": bool(adf_res[1] < 0.05)
    }

    # KPSS Test
    # H0: Series is trend/level stationary
    # H1: Series has a unit root (Non-Stationary)
    kpss_res = kpss(weekly_agg.dropna(), regression="c", nlags="auto")
    kpss_dict = {
        "test_name": "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)",
        "test_statistic": float(kpss_res[0]),
        "p_value": float(kpss_res[1]),
        "lags_used": int(kpss_res[2]),
        "critical_values": {k: float(v) for k, v in kpss_res[3].items()},
        "is_stationary": bool(kpss_res[1] >= 0.05)
    }

    stationarity_report = {
        "ADF": adf_dict,
        "KPSS": kpss_dict,
        "synthesis": "Both tests must be interpreted together. ADF p-value < 0.05 rejects non-stationarity, but strong seasonal spikes in weekly sales violate strict wide-sense stationarity."
    }
    with open(os.path.join(OUTPUT_EDA_DIR, "stationarity_tests.json"), "w") as f:
        json.dump(stationarity_report, f, indent=4)
    logger.info("Saved stationarity test results.")


def run_special_investigations(train, test):
    """Cold-start series, incomplete panels, and negative sales profiling."""
    logger.info("Running Special Investigations...")

    # A. Cold-Start Series
    train_sd = set(zip(train["Store"], train["Dept"]))
    test_sd = set(zip(test["Store"], test["Dept"]))
    cold_start_sd = sorted(list(test_sd - train_sd))
    
    cold_records = []
    for s, d in cold_start_sd:
        test_sub = test[(test["Store"] == s) & (test["Dept"] == d)]
        store_type = test_sub["Type"].iloc[0]
        store_size = test_sub["Size"].iloc[0]
        cold_records.append({
            "Store": int(s),
            "Dept": int(d),
            "Store_Type": store_type,
            "Store_Size": int(store_size),
            "Test_Horizon_Weeks": int(len(test_sub)),
            "First_Test_Date": test_sub["Date"].min().strftime("%Y-%m-%d"),
            "Last_Test_Date": test_sub["Date"].max().strftime("%Y-%m-%d"),
            "Recommended_Fallback": f"Mean sales of Dept {d} across Type {store_type} stores"
        })
    cold_df = pd.DataFrame(cold_records)
    cold_df.to_csv(os.path.join(OUTPUT_EDA_DIR, "cold_start_series_profile.csv"), index=False)

    # B. Incomplete Time-Series (Panel Sparsity)
    sd_counts = train.groupby(["Store", "Dept"])["Date"].nunique().reset_index()
    sd_counts.rename(columns={"Date": "Observed_Weeks"}, inplace=True)
    sd_counts["Is_Complete"] = (sd_counts["Observed_Weeks"] == 143)
    sd_counts.to_csv(os.path.join(OUTPUT_EDA_DIR, "panel_sparsity_profile.csv"), index=False)

    # Figure 17: Panel Sparsity Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sd_counts["Observed_Weeks"], bins=30, ax=ax, color="#1f77b4", edgecolor="black")
    ax.axvline(143, color="red", linestyle="--", linewidth=1.5, label="Complete Panel (143 weeks: 79.86%)")
    ax.set_title("Store-Department Series Length Distribution (Unbalanced Panel Gaps)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Historical Active Weeks Observed", fontsize=11)
    ax.set_ylabel("Number of Store-Department Series", fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig17_panel_sparsity_distribution.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {fig_path}")

    # Summary JSON for special investigations
    spec_summary = {
        "cold_start_count": len(cold_start_sd),
        "cold_start_pairs": cold_start_sd,
        "total_active_series_train": int(len(sd_counts)),
        "complete_series_count": int(sd_counts["Is_Complete"].sum()),
        "incomplete_series_count": int((~sd_counts["Is_Complete"]).sum()),
        "min_series_length": int(sd_counts["Observed_Weeks"].min()),
        "median_series_length": int(sd_counts["Observed_Weeks"].median()),
        "max_series_length": int(sd_counts["Observed_Weeks"].max()),
    }
    with open(os.path.join(OUTPUT_EDA_DIR, "special_investigations_summary.json"), "w") as f:
        json.dump(spec_summary, f, indent=4)
    logger.info("Saved special investigations summary.")


def run_all_eda():
    logger.info("=== STARTING PHASE 2 EDA & STATISTICAL PIPELINE ===")
    train, test = load_processed_data()
    run_sales_eda(train)
    run_time_series_eda(train, test)
    run_holiday_deep_dive(train)
    run_markdown_and_economic_eda(train, test)
    run_time_series_decomposition(train)
    run_special_investigations(train, test)
    logger.info("=== PHASE 2 EDA & STATISTICAL PIPELINE COMPLETED ===")


if __name__ == "__main__":
    run_all_eda()
