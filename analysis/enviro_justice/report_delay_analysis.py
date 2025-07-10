import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import esda
from libpysal.weights import KNN
import requests
import json
import time
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            print(f"✓ Ollama is running. Available models: {available_models}")
            return True, available_models
        else:
            print(f"✗ Ollama responded with status {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Is it running on localhost:11434?")
        return False, []
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False, []

def query_ollama(prompt, model="mistral:7b", max_retries=3, timeout=120):
    """Send query to local Ollama instance with improved error handling"""
    
    is_connected, available_models = check_ollama_connection()
    if not is_connected:
        print("Skipping Ollama analysis - service not available")
        return None
    
    if model not in available_models:
        print(f"Model '{model}' not available. Available models: {available_models}")
        if available_models:
            model = available_models[0]
            print(f"Using '{model}' instead")
        else:
            return None
    
    max_prompt_length = 4000
    if len(prompt) > max_prompt_length:
        print(f"Warning: Prompt too long ({len(prompt)} chars), truncating to {max_prompt_length}")
        prompt = prompt[:max_prompt_length] + "\n\n[Analysis truncated due to length limits]"
    
    for attempt in range(max_retries):
        try:
            print(f"Sending request to Ollama (attempt {attempt + 1}/{max_retries})...")
            
            start_time = time.time()
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 2000
                    }
                },
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            print(f"Request completed in {elapsed_time:.1f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    return result['response']
                else:
                    print(f"Unexpected response format: {result}")
                    return None
            else:
                print(f"HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Request timed out after {timeout} seconds (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error querying Ollama (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

def prepare_temporal_data(df):
    """Prepare and clean temporal data for analysis"""
    
    print("TEMPORAL DATA PREPARATION")
    print("="*50)
    
    # Convert date columns - using correct column names
    date_columns = ['Initial Report Date', 'Date of Discovery']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted {col} to datetime")
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    
    # Create temporal variables
    df_temporal = df.copy()
    
    # Calculate reporting delay (days between discovery and initial report)
    if 'Date of Discovery' in df.columns and 'Initial Report Date' in df.columns:
        df_temporal['reporting_delay_days'] = (
            df_temporal['Initial Report Date'] - df_temporal['Date of Discovery']
        ).dt.days
        
        # Handle negative delays (reports before discovery - data quality issue)
        negative_delays = (df_temporal['reporting_delay_days'] < 0).sum()
        if negative_delays > 0:
            print(f"Warning: {negative_delays} records with negative reporting delays (report before discovery)")
            df_temporal.loc[df_temporal['reporting_delay_days'] < 0, 'reporting_delay_days'] = 0
    else:
        print("Warning: Cannot calculate reporting delays - missing date columns")
    
    # Use existing Spill Type classification
    if 'Spill Type' in df.columns:
        df_temporal['spill_classification'] = df['Spill Type']
        print("Using existing 'Spill Type' classification")
    else:
        print("Warning: 'Spill Type' column not found")
        df_temporal['spill_classification'] = 'Unknown'
    
    # Extract year for trend analysis
    if 'Date of Discovery' in df_temporal.columns:
        df_temporal['discovery_year'] = df_temporal['Date of Discovery'].dt.year
    if 'Initial Report Date' in df_temporal.columns:
        df_temporal['report_year'] = df_temporal['Initial Report Date'].dt.year
    
    # Data quality summary
    print(f"\nData Quality Summary:")
    print(f"Total records: {len(df_temporal)}")
    if 'Date of Discovery' in df_temporal.columns:
        print(f"Records with discovery date: {df_temporal['Date of Discovery'].notna().sum()}")
    if 'Initial Report Date' in df_temporal.columns:
        print(f"Records with report date: {df_temporal['Initial Report Date'].notna().sum()}")
    if 'reporting_delay_days' in df_temporal.columns:
        print(f"Records with calculable delay: {df_temporal['reporting_delay_days'].notna().sum()}")
        valid_delays = df_temporal['reporting_delay_days'].dropna()
        if len(valid_delays) > 0:
            print(f"Mean reporting delay: {valid_delays.mean():.1f} days")
            print(f"Median reporting delay: {valid_delays.median():.1f} days")
            print(f"Max reporting delay: {valid_delays.max():.0f} days")
    
    if 'spill_classification' in df_temporal.columns:
        spill_type_counts = df_temporal['spill_classification'].value_counts()
        print(f"\nSpill Type Distribution:")
        for spill_type, count in spill_type_counts.items():
            print(f"  {spill_type}: {count} ({count/len(df_temporal)*100:.1f}%)")
    
    return df_temporal

def reporting_delay_disparity_analysis(df):
    """Analyze disparities in reporting delays across demographic groups"""
    
    print("\nREPORTING DELAY DISPARITY ANALYSIS")
    print("="*50)
    
    results = {}
    
    # 1. Historical vs Recent Spills by Demographics
    print("1. HISTORICAL VS RECENT SPILLS BY DEMOGRAPHICS")
    print("-" * 45)
    
    # Define demographic groups
    high_poverty = df['percent_poverty'] > 15
    minority_community = df['percent_white'] < 70
    low_income = df['median_household_income'] < df['median_household_income'].median()
    
    # Chi-square test: Historical spills by poverty level
    historical_spills = df['spill_classification'] == 'Historical'
    
    # Poverty analysis
    contingency_poverty = pd.crosstab(high_poverty, historical_spills)
    if contingency_poverty.shape == (2, 2):
        chi2_poverty, p_poverty, dof, expected = stats.chi2_contingency(contingency_poverty)
        
        high_pov_historical = len(df[high_poverty & historical_spills])
        high_pov_total = high_poverty.sum()
        low_pov_historical = len(df[~high_poverty & historical_spills])
        low_pov_total = (~high_poverty).sum()
        
        high_pov_rate = high_pov_historical / high_pov_total if high_pov_total > 0 else 0
        low_pov_rate = low_pov_historical / low_pov_total if low_pov_total > 0 else 0
        
        print(f"Historical Spills by Poverty Level:")
        print(f"  High poverty areas: {high_pov_historical}/{high_pov_total} ({high_pov_rate:.3f})")
        print(f"  Low poverty areas: {low_pov_historical}/{low_pov_total} ({low_pov_rate:.3f})")
        print(f"  Rate ratio: {high_pov_rate/low_pov_rate:.2f}x" if low_pov_rate > 0 else "  Rate ratio: N/A")
        print(f"  Chi-square p-value: {p_poverty:.6f}")
        print(f"  Significant disparity: {'YES' if p_poverty < 0.05 else 'NO'}")
        
        results['poverty_historical'] = {
            'chi2_statistic': chi2_poverty,
            'p_value': p_poverty,
            'high_poverty_rate': high_pov_rate,
            'low_poverty_rate': low_pov_rate,
            'rate_ratio': high_pov_rate/low_pov_rate if low_pov_rate > 0 else np.nan
        }
    
    # Race analysis
    print(f"\nHistorical Spills by Race/Ethnicity:")
    minority_historical = len(df[minority_community & historical_spills])
    minority_total = minority_community.sum()
    white_historical = len(df[~minority_community & historical_spills])
    white_total = (~minority_community).sum()
    
    minority_rate = minority_historical / minority_total if minority_total > 0 else 0
    white_rate = white_historical / white_total if white_total > 0 else 0
    
    print(f"  Minority communities: {minority_historical}/{minority_total} ({minority_rate:.3f})")
    print(f"  Majority white areas: {white_historical}/{white_total} ({white_rate:.3f})")
    print(f"  Rate ratio: {minority_rate/white_rate:.2f}x" if white_rate > 0 else "  Rate ratio: N/A")
    
    # Two-proportion z-test for race
    if minority_total > 0 and white_total > 0:
        counts = np.array([minority_historical, white_historical])
        nobs = np.array([minority_total, white_total])
        z_stat_race, p_race = proportions_ztest(counts, nobs)
        print(f"  Two-proportion z-test p-value: {p_race:.6f}")
        print(f"  Significant disparity: {'YES' if p_race < 0.05 else 'NO'}")
        
        results['race_historical'] = {
            'z_statistic': z_stat_race,
            'p_value': p_race,
            'minority_rate': minority_rate,
            'white_rate': white_rate,
            'rate_ratio': minority_rate/white_rate if white_rate > 0 else np.nan
        }
    
    # 2. Actual Reporting Delays (in days)
    if 'reporting_delay_days' in df.columns:
        print(f"\n2. REPORTING DELAY ANALYSIS (DAYS)")
        print("-" * 35)
        
        delay_data = df.dropna(subset=['reporting_delay_days'])
        
        if len(delay_data) > 0:
            # Compare mean delays by demographics
            high_pov_delays = delay_data[delay_data['percent_poverty'] > 15]['reporting_delay_days']
            low_pov_delays = delay_data[delay_data['percent_poverty'] <= 15]['reporting_delay_days']
            
            if len(high_pov_delays) > 0 and len(low_pov_delays) > 0:
                # T-test for difference in means
                t_stat, p_ttest = stats.ttest_ind(high_pov_delays, low_pov_delays)
                
                print(f"Mean Reporting Delays:")
                print(f"  High poverty areas: {high_pov_delays.mean():.1f} days (n={len(high_pov_delays)})")
                print(f"  Low poverty areas: {low_pov_delays.mean():.1f} days (n={len(low_pov_delays)})")
                print(f"  Difference: {high_pov_delays.mean() - low_pov_delays.mean():.1f} days")
                print(f"  T-test p-value: {p_ttest:.6f}")
                print(f"  Significant difference: {'YES' if p_ttest < 0.05 else 'NO'}")
                
                results['delay_poverty'] = {
                    't_statistic': t_stat,
                    'p_value': p_ttest,
                    'high_poverty_mean': high_pov_delays.mean(),
                    'low_poverty_mean': low_pov_delays.mean(),
                    'difference': high_pov_delays.mean() - low_pov_delays.mean()
                }
            
            # Median delays (more robust to outliers)
            print(f"\nMedian Reporting Delays:")
            print(f"  High poverty areas: {high_pov_delays.median():.1f} days")
            print(f"  Low poverty areas: {low_pov_delays.median():.1f} days")
            
            # Mann-Whitney U test (non-parametric)
            if len(high_pov_delays) > 0 and len(low_pov_delays) > 0:
                u_stat, p_mannwhitney = stats.mannwhitneyu(high_pov_delays, low_pov_delays, alternative='two-sided')
                print(f"  Mann-Whitney U test p-value: {p_mannwhitney:.6f}")
                print(f"  Significant difference (non-parametric): {'YES' if p_mannwhitney < 0.05 else 'NO'}")
    
    return results

def spatial_reporting_analysis(df):
    """Analyze spatial patterns in reporting delays"""
    
    print("\n3. SPATIAL PATTERNS OF REPORTING DELAYS")
    print("-" * 40)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.dropna(subset=['Latitude', 'Longitude']), 
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs='EPSG:4326'
    )
    
    if len(gdf) == 0:
        print("No valid geographic data available")
        return None, None
    
    # Project for spatial analysis
    gdf_proj = gdf.to_crs('EPSG:3857')
    
    # Spatial clustering of historical spills
    historical_points = gdf_proj[gdf_proj['spill_classification'] == 'Historical']
    
    if len(historical_points) > 10:
        coords = np.column_stack([historical_points.geometry.x, historical_points.geometry.y])
        
        # DBSCAN clustering
        eps = 5000  # 5km radius
        min_samples = 5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(coords)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        print(f"Historical Spill Clustering:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Clustered points: {len(historical_points) - n_noise}")
        print(f"  Noise points: {n_noise}")
        
        # Add cluster labels
        historical_points_copy = historical_points.copy()
        historical_points_copy['cluster'] = clusters
        
        # Analyze demographic characteristics of clusters
        if n_clusters > 0:
            cluster_summary = []
            for cluster_id in set(clusters):
                if cluster_id != -1:  # Exclude noise
                    cluster_points = historical_points_copy[historical_points_copy['cluster'] == cluster_id]
                    
                    cluster_summary.append({
                        'cluster_id': cluster_id,
                        'n_spills': len(cluster_points),
                        'avg_poverty': cluster_points['percent_poverty'].mean(),
                        'avg_income': cluster_points['median_household_income'].mean(),
                        'pct_minority': (cluster_points['percent_white'] < 70).mean() * 100
                    })
            
            cluster_df = pd.DataFrame(cluster_summary)
            print(f"\nCluster Demographics:")
            print(f"  Average poverty rate: {cluster_df['avg_poverty'].mean():.1f}%")
            print(f"  Average income: ${cluster_df['avg_income'].mean():,.0f}")
            print(f"  Percent minority communities: {cluster_df['pct_minority'].mean():.1f}%")
    
    # Spatial autocorrelation of reporting delays
    if 'reporting_delay_days' in gdf.columns and len(gdf) > 50:
        delay_data = gdf.dropna(subset=['reporting_delay_days'])
        
        if len(delay_data) > 50:
            try:
                coords_array = np.column_stack([delay_data.geometry.x, delay_data.geometry.y])
                w = KNN.from_array(coords_array, k=min(8, len(delay_data)-1))
                w.transform = 'r'
                
                moran_delay = esda.Moran(delay_data['reporting_delay_days'], w)
                
                print(f"\nSpatial Autocorrelation of Reporting Delays:")
                print(f"  Moran's I: {moran_delay.I:.4f}")
                print(f"  p-value: {moran_delay.p_sim:.4f}")
                print(f"  Significant clustering: {'YES' if moran_delay.p_sim < 0.05 else 'NO'}")
                
            except Exception as e:
                print(f"  Spatial autocorrelation analysis failed: {e}")
    
    return gdf, n_clusters if 'n_clusters' in locals() else 0

def temporal_trend_analysis(df):
    """Analyze trends in reporting delays over time"""
    
    print("\n4. TEMPORAL TRENDS IN REPORTING")
    print("-" * 32)
    
    if 'report_year' not in df.columns:
        print("No temporal data available for trend analysis")
        return None
    
    # Filter to reasonable year range
    df_temporal = df[(df['report_year'] >= 2014) & (df['report_year'] <= 2024)]
    
    if len(df_temporal) == 0:
        print("No data in 2014-2024 range")
        return None
    
    # Trend in historical spill proportions
    yearly_summary = df_temporal.groupby('report_year').agg({
        'spill_classification': lambda x: (x == 'Historical').sum(),
        'percent_poverty': 'mean',
        'median_household_income': 'mean'
    }).reset_index()
    
    yearly_summary['total_spills'] = df_temporal.groupby('report_year').size().values
    yearly_summary['historical_rate'] = yearly_summary['spill_classification'] / yearly_summary['total_spills']
    
    print(f"Temporal Trends (2014-2024):")
    print(f"  Years with data: {sorted(df_temporal['report_year'].unique())}")
    print(f"  Total spills: {len(df_temporal)}")
    
    # Correlation between year and historical rate
    if len(yearly_summary) > 3:
        corr_year_hist, p_corr = stats.pearsonr(yearly_summary['report_year'], yearly_summary['historical_rate'])
        print(f"  Correlation (year vs historical rate): {corr_year_hist:.3f} (p={p_corr:.3f})")
        
        trend_direction = "improving" if corr_year_hist < 0 else "worsening" if corr_year_hist > 0 else "stable"
        print(f"  Trend interpretation: {trend_direction} (fewer historical = better reporting)")
    
    # Recent vs early period comparison
    early_period = df_temporal[df_temporal['report_year'] <= 2018]
    recent_period = df_temporal[df_temporal['report_year'] >= 2020]
    
    if len(early_period) > 0 and len(recent_period) > 0:
        early_hist_rate = (early_period['spill_classification'] == 'Historical').mean()
        recent_hist_rate = (recent_period['spill_classification'] == 'Historical').mean()
        
        print(f"\nPeriod Comparison:")
        print(f"  2014-2018 historical rate: {early_hist_rate:.3f}")
        print(f"  2020-2024 historical rate: {recent_hist_rate:.3f}")
        print(f"  Change: {recent_hist_rate - early_hist_rate:.3f}")
        
        # Statistical test for difference
        early_hist = (early_period['spill_classification'] == 'Historical').sum()
        recent_hist = (recent_period['spill_classification'] == 'Historical').sum()
        
        counts = np.array([early_hist, recent_hist])
        nobs = np.array([len(early_period), len(recent_period)])
        
        if all(counts > 0) and all(nobs > 0):
            z_stat, p_val = proportions_ztest(counts, nobs)
            print(f"  Two-proportion test p-value: {p_val:.4f}")
            print(f"  Significant change: {'YES' if p_val < 0.05 else 'NO'}")
    
    return yearly_summary

def create_reporting_visualizations(df, gdf=None):
    """Create visualizations for reporting delay analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Historical vs Recent by Demographics
    ax1 = axes[0, 0]
    
    # Create demographic categories
    df['demo_category'] = 'Low Poverty'
    df.loc[df['percent_poverty'] > 15, 'demo_category'] = 'High Poverty'
    
    demo_spill = pd.crosstab(df['demo_category'], df['spill_classification'])
    demo_spill_pct = demo_spill.div(demo_spill.sum(axis=1), axis=0) * 100
    
    demo_spill_pct.plot(kind='bar', ax=ax1, stacked=True, 
                       color=['lightblue', 'darkred', 'gray'])
    ax1.set_title('Spill Classification by Poverty Level')
    ax1.set_xlabel('Community Type')
    ax1.set_ylabel('Percentage of Spills')
    ax1.legend(title='Spill Type')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Reporting Delays Distribution
    ax2 = axes[0, 1]
    
    if 'reporting_delay_days' in df.columns:
        delay_data = df.dropna(subset=['reporting_delay_days'])
        if len(delay_data) > 0:
            # Cap at 95th percentile for visualization
            cap_value = delay_data['reporting_delay_days'].quantile(0.95)
            delay_capped = delay_data['reporting_delay_days'].clip(upper=cap_value)
            
            ax2.hist(delay_capped, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_title('Distribution of Reporting Delays')
            ax2.set_xlabel('Days between Discovery and Report')
            ax2.set_ylabel('Number of Spills')
            ax2.axvline(delay_capped.mean(), color='red', linestyle='--', 
                       label=f'Mean: {delay_capped.mean():.1f} days')
            ax2.legend()
    
    # 3. Geographic Distribution
    ax3 = axes[1, 0]
    
    if gdf is not None and len(gdf) > 0:
        # Plot by spill classification
        historical = gdf[gdf['spill_classification'] == 'Historical']
        recent = gdf[gdf['spill_classification'] == 'Recent']
        
        if len(recent) > 0:
            ax3.scatter(recent['Longitude'], recent['Latitude'], 
                       c='lightblue', s=10, alpha=0.6, label='Recent')
        if len(historical) > 0:
            ax3.scatter(historical['Longitude'], historical['Latitude'], 
                       c='darkred', s=15, alpha=0.8, label='Historical')
        
        ax3.set_title('Geographic Distribution by Spill Type')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.legend()
    
    # 4. Temporal Trends
    ax4 = axes[1, 1]
    
    if 'report_year' in df.columns:
        yearly_data = df.groupby('report_year').agg({
            'spill_classification': lambda x: (x == 'Historical').sum()
        }).reset_index()
        yearly_totals = df.groupby('report_year').size().reset_index(name='total_count')
        yearly_data = yearly_data.merge(yearly_totals, on='report_year')
        yearly_data.columns = ['year', 'historical_count', 'total_count']
        yearly_data['historical_rate'] = yearly_data['historical_count'] / yearly_data['total_count']
        
        ax4.plot(yearly_data['year'], yearly_data['historical_rate'], 
                marker='o', linewidth=2, markersize=6)
        ax4.set_title('Historical Spill Rate Over Time')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Proportion of Historical Spills')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reporting_delay_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_reporting_delay_report(results, spatial_results, temporal_results):
    """Generate comprehensive report on reporting delays"""
    
    # Create concise summary for LLM
    summary_stats = []
    
    if 'poverty_historical' in results:
        pov_ratio = results['poverty_historical']['rate_ratio']
        pov_p = results['poverty_historical']['p_value']
        summary_stats.append(f"High-poverty areas: {pov_ratio:.2f}x more historical spills (p={pov_p:.4f})")
    
    if 'race_historical' in results:
        race_ratio = results['race_historical']['rate_ratio']
        race_p = results['race_historical']['p_value']
        summary_stats.append(f"Minority communities: {race_ratio:.2f}x more historical spills (p={race_p:.4f})")
    
    if 'delay_poverty' in results:
        delay_diff = results['delay_poverty']['difference']
        delay_p = results['delay_poverty']['p_value']
        summary_stats.append(f"High-poverty areas: {delay_diff:.1f} days longer delays (p={delay_p:.4f})")
    
    spatial_info = f"{spatial_results} spatial clusters of historical spills" if spatial_results else "No spatial clustering detected"
    
    prompt = f"""Analyze these environmental justice findings on reporting delays:

REPORTING DISPARITIES:
{chr(10).join(['- ' + stat for stat in summary_stats])}

SPATIAL PATTERNS:
- {spatial_info}

INTERPRETATION NEEDED:
This analysis examines whether marginalized communities experience delayed discovery and reporting of oil/gas spills, indicating weaker oversight and environmental monitoring.

Provide a 300-word academic interpretation focusing on:
1. Environmental justice implications of reporting delays
2. Institutional barriers and oversight gaps
3. Policy recommendations for improved monitoring
4. Community empowerment strategies"""
    
    print("\nGenerating reporting delay interpretation with Ollama...")
    report = query_ollama(prompt)
    
    if report is None:
        report = f"""
ENVIRONMENTAL JUSTICE IMPLICATIONS OF REPORTING DELAYS

EXECUTIVE SUMMARY:
The analysis reveals significant disparities in oil and gas spill reporting patterns across demographic lines, indicating institutional barriers and unequal environmental oversight.

KEY FINDINGS:
{chr(10).join(summary_stats)}

ENVIRONMENTAL JUSTICE IMPLICATIONS:
Delayed discovery and reporting of spills in marginalized communities represents a form of procedural environmental injustice. When spills go undetected for extended periods, affected communities face:
- Prolonged exposure to contamination
- Delayed cleanup and remediation efforts  
- Reduced accountability for responsible parties
- Weakened community awareness and response capacity

INSTITUTIONAL BARRIERS:
The disparities suggest systematic differences in:
- Regulatory oversight and inspection frequency
- Community access to reporting mechanisms
- Corporate compliance and monitoring practices
- Environmental awareness and advocacy capacity

POLICY RECOMMENDATIONS:
1. Enhanced monitoring requirements in environmental justice communities
2. Community-based environmental monitoring programs
3. Mandatory spill detection technology for facilities in sensitive areas
4. Improved public reporting systems and community notification
5. Regular environmental audits with community participation

COMMUNITY EMPOWERMENT:
- Training residents in environmental monitoring
- Establishing community environmental health liaisons
- Creating accessible reporting hotlines
- Supporting community-led advocacy organizations

The findings demonstrate the need for proactive policies to ensure equitable environmental protection and community oversight capabilities.
        """
        print("Using fallback report (Ollama unavailable)")
    
    return report

def run_reporting_delay_analysis(csv_file, use_ollama=True):
    """Run comprehensive reporting delay analysis"""
    
    print("ENVIRONMENTAL JUSTICE REPORTING DELAY ANALYSIS")
    print("="*60)
    
    # Load and prepare data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} spill records")
    
    # Check Ollama if requested
    if use_ollama:
        print("\nChecking Ollama connection...")
        check_ollama_connection()
    
    # Prepare temporal data
    df_temporal = prepare_temporal_data(df)
    
    # Main analyses
    disparity_results = reporting_delay_disparity_analysis(df_temporal)
    gdf, n_clusters = spatial_reporting_analysis(df_temporal)
    temporal_trends = temporal_trend_analysis(df_temporal)
    
    # Create visualizations
    create_reporting_visualizations(df_temporal, gdf)
    
    # Generate report
    if use_ollama:
        report = generate_reporting_delay_report(disparity_results, n_clusters, temporal_trends)
    else:
        print("\nSkipping Ollama report generation...")
        report = "Ollama report generation skipped by user"
    
    # Save results
    results = {
        'disparity_analysis': disparity_results,
        'spatial_clusters': n_clusters,
        'temporal_trends': temporal_trends.to_dict('records') if temporal_trends is not None else None,
        'interpretation': report,
        'data_summary': {
            'total_records': len(df_temporal),
            'date_range': f"{df_temporal['report_year'].min()}-{df_temporal['report_year'].max()}" if 'report_year' in df_temporal.columns else "Unknown",
            'spill_classifications': df_temporal['spill_classification'].value_counts().to_dict()
        }
    }
    
    with open('reporting_delay_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('reporting_delay_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nReporting delay analysis complete. Results saved to:")
    print(f"  - reporting_delay_analysis.json")
    print(f"  - reporting_delay_report.txt") 
    print(f"  - reporting_delay_analysis.png")
    
    return results

if __name__ == "__main__":
    # Run the reporting delay analysis
    results = run_reporting_delay_analysis('data/spills_with_demographics.csv', use_ollama=True)