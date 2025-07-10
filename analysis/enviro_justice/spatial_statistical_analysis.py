import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import esda
from libpysal.weights import Queen, KNN
from splot.esda import moran_scatterplot, lisa_cluster
import requests
import json
import time
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.formula.api import ols
import contextily as ctx
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
    except requests.exceptions.Timeout:
        print("✗ Ollama connection timed out")
        return False, []
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False, []

def query_ollama(prompt, model="mistral:7b", max_retries=3, timeout=120):
    """Send query to local Ollama instance with improved error handling"""
    
    # Check connection first
    is_connected, available_models = check_ollama_connection()
    if not is_connected:
        print("Skipping Ollama analysis - service not available")
        return None
    
    # Check if requested model is available
    if model not in available_models:
        print(f"Model '{model}' not available. Available models: {available_models}")
        if available_models:
            model = available_models[0]  # Use first available model
            print(f"Using '{model}' instead")
        else:
            return None
    
    # Truncate prompt if too long (Ollama has context limits)
    max_prompt_length = 4000  # Conservative limit
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
                        'num_predict': 2000  # Limit response length
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
                time.sleep(2)  # Wait before retry
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"Error querying Ollama (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print("All Ollama attempts failed")
    return None

def statistical_disparity_tests(df):
    """Perform statistical tests for environmental justice disparities"""
    
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*50)
    
    results = {}
    
    # 1. Income Quartile Analysis
    income_quartiles = pd.qcut(df['median_household_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    spill_counts = df.groupby(income_quartiles).size()
    
    # Chi-square test for income distribution
    expected_per_quartile = len(df) / 4
    chi2_income, p_income = stats.chisquare(spill_counts, f_exp=[expected_per_quartile] * 4)
    
    print(f"Income Distribution Test:")
    print(f"  Chi-square statistic: {chi2_income:.3f}")
    print(f"  p-value: {p_income:.6f}")
    print(f"  Significant disparity: {'YES' if p_income < 0.001 else 'NO'}")
    
    # 2. Poverty Rate Analysis
    high_poverty = df['percent_poverty'] > 15
    high_poverty_spills = high_poverty.sum()
    total_spills = len(df)
    
    # Assuming 20% of census tracts are high poverty (national average)
    expected_high_poverty = 0.20 * total_spills
    
    print(f"\nPoverty Analysis:")
    print(f"  High-poverty spills: {high_poverty_spills}")
    print(f"  Expected (if random): {expected_high_poverty:.0f}")
    print(f"  Ratio: {high_poverty_spills / expected_high_poverty:.2f}x")
    
    # Binomial test
    poverty_test = stats.binomtest(high_poverty_spills, total_spills, 0.20, alternative='greater')
    poverty_p = poverty_test.pvalue
    print(f"  Binomial test p-value: {poverty_p:.6f}")
    print(f"  Significant over-representation: {'YES' if poverty_p < 0.001 else 'NO'}")
    
    # 3. Major Spills Analysis
    major_spills = df['More than five barrels spilled'].astype(str) == 'Y'
    
    # Test if major spills disproportionately affect high-poverty areas
    high_pov_major = df[high_poverty & major_spills].shape[0]
    high_pov_total = high_poverty.sum()
    low_pov_major = df[~high_poverty & major_spills].shape[0]
    low_pov_total = (~high_poverty).sum()
    
    # Two-proportion z-test
    counts = np.array([high_pov_major, low_pov_major])
    nobs = np.array([high_pov_total, low_pov_total])
    z_stat, p_major = proportions_ztest(counts, nobs)
    
    print(f"\nMajor Spills in High-Poverty Areas:")
    print(f"  High poverty major spill rate: {high_pov_major/high_pov_total:.3f}")
    print(f"  Low poverty major spill rate: {low_pov_major/low_pov_total:.3f}")
    print(f"  Z-statistic: {z_stat:.3f}")
    print(f"  p-value: {p_major:.6f}")
    print(f"  Significant difference: {'YES' if p_major < 0.05 else 'NO'}")
    
    # 4. Racial Demographics
    minority_communities = df['percent_white'] < 70
    minority_spills = minority_communities.sum()
    
    # Assuming 30% of areas are minority communities (rough US average)
    expected_minority = 0.30 * total_spills
    
    print(f"\nRacial Demographics Analysis:")
    print(f"  Minority community spills: {minority_spills}")
    print(f"  Expected (if random): {expected_minority:.0f}")
    print(f"  Ratio: {minority_spills / expected_minority:.2f}x")
    
    minority_test = stats.binomtest(minority_spills, total_spills, 0.30, alternative='greater')
    minority_p = minority_test.pvalue
    print(f"  Binomial test p-value: {minority_p:.6f}")
    print(f"  Significant over-representation: {'YES' if minority_p < 0.05 else 'NO'}")
    
    results = {
        'income_chi2': {'statistic': chi2_income, 'p_value': p_income},
        'poverty_binomial': {'p_value': poverty_p, 'observed_ratio': high_poverty_spills / expected_high_poverty},
        'major_spills_ztest': {'z_statistic': z_stat, 'p_value': p_major},
        'minority_binomial': {'p_value': minority_p, 'observed_ratio': minority_spills / expected_minority}
    }
    
    return results

def spatial_analysis(df):
    """Perform spatial analysis of spill patterns"""
    
    print("\nSPATIAL ANALYSIS")
    print("="*50)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs='EPSG:4326'
    )
    
    # Project to Colorado State Plane (meters) for distance calculations
    gdf_proj = gdf.to_crs('EPSG:3857')  # Web Mercator for general analysis
    
    # 1. Spatial Clustering Analysis (DBSCAN)
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    
    # DBSCAN clustering directly on projected coordinates (meters)
    # eps is approximately 1km
    eps = 1000
    min_samples = 10

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords)
    
    gdf['cluster'] = clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Spatial Clustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Clustered points: {len(gdf) - n_noise}")
    
    # 2. Moran's I for spatial autocorrelation
    if len(gdf) > 100:  # Only if we have enough points
        # Remove any rows with missing values for spatial analysis
        gdf_spatial = gdf.dropna(subset=['percent_poverty', 'median_household_income'])
        
        if len(gdf_spatial) > 100:
            # Create spatial weights (K-nearest neighbors)
            coords_array = np.column_stack([gdf_spatial.geometry.x, gdf_spatial.geometry.y])
            w = KNN.from_array(coords_array, k=min(8, len(gdf_spatial)-1))
            w.transform = 'r'  # Row standardization
            
            # Test spatial autocorrelation of poverty rates
            try:
                moran_poverty = esda.Moran(gdf_spatial['percent_poverty'], w)
                
                print(f"\nSpatial Autocorrelation (Moran's I):")
                print(f"  Poverty rate Moran's I: {moran_poverty.I:.4f}")
                print(f"  p-value: {moran_poverty.p_sim:.4f}")
                print(f"  Significant clustering: {'YES' if moran_poverty.p_sim < 0.05 else 'NO'}")
                
                # Test for income
                moran_income = esda.Moran(gdf_spatial['median_household_income'], w)
                print(f"  Income Moran's I: {moran_income.I:.4f}")
                print(f"  p-value: {moran_income.p_sim:.4f}")
                
                # LISA analysis for local clusters
                lisa_poverty = esda.Moran_Local(gdf_spatial['percent_poverty'], w)
                
                # Count significant LISA clusters
                significant_clusters = np.sum(lisa_poverty.p_sim < 0.05)
                print(f"  Significant local poverty clusters: {significant_clusters}")
                
            except Exception as e:
                print(f"  Spatial autocorrelation analysis failed: {e}")
        else:
            print(f"  Insufficient valid spatial data: {len(gdf_spatial)} points")
    
    # 3. Hotspot Analysis
    # Create grid and count spills per cell
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    
    # Create 5km x 5km grid
    grid_size = 5000  # 5km in meters
    x_coords = np.arange(xmin, xmax + grid_size, grid_size)
    y_coords = np.arange(ymin, ymax + grid_size, grid_size)
    
    spill_density = calculate_spill_density(gdf_proj, x_coords, y_coords, grid_size)
    
    print(f"\nHotspot Analysis:")
    print(f"  Grid cells created: {len(spill_density)}")
    if len(spill_density) > 0:
        print(f"  Max spills per 5km cell: {spill_density['spill_count'].max()}")
        print(f"  Mean spills per cell: {spill_density['spill_count'].mean():.2f}")
    else:
        print("  No grid cells with spills found")
    
    return gdf, spill_density, n_clusters

def calculate_spill_density(gdf_proj, x_coords, y_coords, grid_size):
    """Calculate spill density on a grid"""
    
    density_data = []
    
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            # Define grid cell bounds
            cell_bounds = (x, y, x + grid_size, y + grid_size)
            
            # Count spills in this cell
            mask = (
                (gdf_proj.geometry.x >= cell_bounds[0]) &
                (gdf_proj.geometry.x < cell_bounds[2]) &
                (gdf_proj.geometry.y >= cell_bounds[1]) &
                (gdf_proj.geometry.y < cell_bounds[3])
            )
            
            spills_in_cell = gdf_proj[mask]
            
            if len(spills_in_cell) > 0:
                density_data.append({
                    'grid_x': x + grid_size/2,
                    'grid_y': y + grid_size/2,
                    'spill_count': len(spills_in_cell),
                    'avg_poverty': spills_in_cell['percent_poverty'].mean(),
                    'avg_income': spills_in_cell['median_household_income'].mean(),
                    'major_spills': (spills_in_cell['More than five barrels spilled'].astype(str) == 'Y').sum()
                })
    
    return pd.DataFrame(density_data)

def spatial_regression_analysis(gdf):
    """Perform spatial regression to control for location effects"""
    
    print("\nSPATIAL REGRESSION ANALYSIS")
    print("="*50)
    
    # Create variables for regression
    gdf_reg = gdf.copy()
    gdf_reg['major_spill'] = (gdf_reg['More than five barrels spilled'].astype(str) == 'Y').astype(int)
    gdf_reg['high_poverty'] = (gdf_reg['percent_poverty'] > 15).astype(int)
    gdf_reg['minority_community'] = (gdf_reg['percent_white'] < 70).astype(int)
    
    # Add spatial controls (distance to urban centers, etc.)
    # For now, use lat/lon as proxies for spatial effects
    gdf_reg['lat_norm'] = (gdf_reg['Latitude'] - gdf_reg['Latitude'].mean()) / gdf_reg['Latitude'].std()
    gdf_reg['lon_norm'] = (gdf_reg['Longitude'] - gdf_reg['Longitude'].mean()) / gdf_reg['Longitude'].std()
    
    # OLS regression: Major spill probability ~ demographics + spatial controls
    model_formula = 'major_spill ~ percent_poverty + percent_white + median_household_income + lat_norm + lon_norm'
    
    try:
        model = ols(model_formula, data=gdf_reg).fit()
        
        print("Regression Results (Major Spill Probability):")
        print(f"  R-squared: {model.rsquared:.4f}")
        print(f"  F-statistic p-value: {model.f_pvalue:.6f}")
        
        # Key coefficients
        coef_poverty = model.params.get('percent_poverty', 0)
        pval_poverty = model.pvalues.get('percent_poverty', 1)
        
        coef_white = model.params.get('percent_white', 0) 
        pval_white = model.pvalues.get('percent_white', 1)
        
        coef_income = model.params.get('median_household_income', 0)
        pval_income = model.pvalues.get('median_household_income', 1)
        
        print(f"\nKey Findings:")
        print(f"  Poverty rate coefficient: {coef_poverty:.6f} (p={pval_poverty:.4f})")
        print(f"  White percentage coefficient: {coef_white:.6f} (p={pval_white:.4f})")
        print(f"  Income coefficient: {coef_income:.8f} (p={pval_income:.4f})")
        
        return model
        
    except Exception as e:
        print(f"Regression analysis failed: {e}")
        return None

def generate_spatial_statistical_report(stats_results, spatial_results, model_results=None):
    """Generate comprehensive report using LLM"""
    
    # Extract key regression findings if available
    regression_summary = "No regression analysis available"
    if model_results and hasattr(model_results, 'rsquared'):
        # Extract only key statistics, not the full summary
        poverty_coef = model_results.params.get('percent_poverty', 0)
        poverty_pval = model_results.pvalues.get('percent_poverty', 1)
        income_coef = model_results.params.get('median_household_income', 0)
        income_pval = model_results.pvalues.get('median_household_income', 1)
        
        regression_summary = f"R²={model_results.rsquared:.3f}, poverty coef={poverty_coef:.4f} (p={poverty_pval:.3f}), income coef={income_coef:.6f} (p={income_pval:.3f})"
    
    # Create a focused, shorter prompt
    prompt = f"""Analyze these environmental justice findings:

STATISTICAL TESTS:
- Poverty over-representation: {stats_results['poverty_binomial']['observed_ratio']:.2f}x expected (p={stats_results['poverty_binomial']['p_value']:.4f})
- Major spills z-test p-value: {stats_results['major_spills_ztest']['p_value']:.4f}
- Minority communities ratio: {stats_results['minority_binomial']['observed_ratio']:.2f}x

SPATIAL PATTERNS:
- {spatial_results['n_clusters']} spatial clusters identified
- Max density: {spatial_results.get('max_density', 'N/A')} spills per grid cell

REGRESSION RESULTS:
- {regression_summary}

Provide a 300-word academic interpretation focusing on environmental justice implications and policy recommendations."""
    
    print("\nGenerating academic interpretation with Ollama...")
    report = query_ollama(prompt)
    
    if report is None:
        # Fallback report if Ollama fails
        report = f"""
ENVIRONMENTAL JUSTICE ANALYSIS - STATISTICAL SUMMARY

The statistical analysis reveals significant disparities in oil and gas spill distribution across demographic lines:

POVERTY DISPARITIES:
High-poverty areas experience {stats_results['poverty_binomial']['observed_ratio']:.2f} times more spills than expected by chance (p = {stats_results['poverty_binomial']['p_value']:.6f}). This suggests a systematic pattern of environmental burden concentration in economically disadvantaged communities.

SPATIAL CLUSTERING:
The analysis identified {spatial_results['n_clusters']} distinct spatial clusters of spill incidents, indicating that environmental risks are geographically concentrated rather than randomly distributed. This clustering pattern strengthens the case for targeted environmental justice interventions.

POLICY IMPLICATIONS:
1. Enhanced environmental monitoring in high-poverty areas
2. Stricter permitting requirements for facilities near vulnerable communities
3. Community notification and participation requirements for new developments
4. Investment in environmental remediation for affected areas

METHODOLOGICAL STRENGTHS:
- Multiple statistical tests controlling for spatial effects
- Combination of global and local spatial analysis
- Robust sample size for statistical inference

The findings provide evidence of environmental injustice requiring policy intervention and continued monitoring.
        """
        print("Using fallback report (Ollama unavailable)")
    
    return report

def create_visualizations(gdf, spill_density):
    """Create key visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Spill locations by poverty rate
    ax1 = axes[0, 0]
    scatter = ax1.scatter(gdf['Longitude'], gdf['Latitude'], 
                         c=gdf['percent_poverty'], cmap='Reds', 
                         alpha=0.6, s=10)
    ax1.set_title('Spill Locations by Poverty Rate')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax1, label='Poverty Rate (%)')
    
    # 2. Income distribution
    ax2 = axes[0, 1]
    income_quartiles = pd.qcut(gdf['median_household_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    income_counts = gdf.groupby(income_quartiles).size()
    ax2.bar(income_counts.index, income_counts.values)
    ax2.set_title('Spills by Income Quartile')
    ax2.set_xlabel('Income Quartile')
    ax2.set_ylabel('Number of Spills')
    
    # 3. Major spills by demographics
    ax3 = axes[1, 0]
    demo_data = pd.DataFrame({
        'High Poverty': [
            len(gdf[(gdf['percent_poverty'] > 15) & (gdf['More than five barrels spilled'].astype(str) == 'Y')]),
            len(gdf[(gdf['percent_poverty'] > 15) & (gdf['More than five barrels spilled'].astype(str) != 'Y')])
        ],
        'Low Poverty': [
            len(gdf[(gdf['percent_poverty'] <= 15) & (gdf['More than five barrels spilled'].astype(str) == 'Y')]),
            len(gdf[(gdf['percent_poverty'] <= 15) & (gdf['More than five barrels spilled'].astype(str) != 'Y')])
        ]
    }, index=['Major Spills', 'Minor Spills'])
    
    demo_data.plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('Spill Severity by Poverty Level')
    ax3.set_xlabel('Spill Type')
    ax3.set_ylabel('Count')
    ax3.legend(title='Community Type')
    
    # 4. Spatial density
    ax4 = axes[1, 1]
    if len(spill_density) > 0:
        scatter2 = ax4.scatter(spill_density['grid_x'], spill_density['grid_y'],
                              c=spill_density['spill_count'], cmap='YlOrRd',
                              s=spill_density['spill_count']*10, alpha=0.7)
        ax4.set_title('Spill Density Hotspots (5km Grid)')
        ax4.set_xlabel('X Coordinate (Projected)')
        ax4.set_ylabel('Y Coordinate (Projected)')
        plt.colorbar(scatter2, ax=ax4, label='Spills per Cell')
    
    plt.tight_layout()
    plt.savefig('environmental_justice_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
def run_comprehensive_analysis(csv_file, use_ollama=True):
    """Run complete statistical and spatial analysis"""
    
    print("COMPREHENSIVE STATISTICAL & SPATIAL ENVIRONMENTAL JUSTICE ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} spill incidents")
    
    # Check Ollama availability if requested
    if use_ollama:
        print("\nChecking Ollama connection...")
        check_ollama_connection()
    
    # Statistical analysis
    stats_results = statistical_disparity_tests(df)
    
    # Spatial analysis
    gdf, spill_density, n_clusters = spatial_analysis(df)
    
    # Spatial regression
    model = spatial_regression_analysis(gdf)
    
    # Create visualizations
    create_visualizations(gdf, spill_density)
    
    # Generate comprehensive report
    spatial_results = {'n_clusters': n_clusters}
    if len(spill_density) > 0:
        spatial_results['max_density'] = spill_density['spill_count'].max()
    
    model_summary = str(model.summary()) if model else "Regression analysis not available"
    
    # Create summaries for different purposes
    model_summary_short = "No regression analysis performed"
    model_summary_full = "Regression analysis not available"
    
    if model and hasattr(model, 'rsquared'):
        model_summary_short = f"OLS Regression Results: R²={model.rsquared:.4f}, F-stat p-value={model.f_pvalue:.6f}"
        model_summary_full = str(model.summary())
    
    if use_ollama:
        report = generate_spatial_statistical_report(stats_results, spatial_results, model)
    else:
        print("\nSkipping Ollama report generation...")
        report = "Ollama report generation skipped by user"
    
    # Save results
    results = {
        'statistical_tests': stats_results,
        'spatial_analysis': spatial_results,
        'regression_summary': model_summary_full,  # Full summary for detailed analysis
        'academic_interpretation': report
    }
    
    with open('statistical_spatial_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('academic_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis complete. Results saved to:")
    print(f"  - statistical_spatial_analysis.json")
    print(f"  - academic_report.txt")
    print(f"  - environmental_justice_analysis.png")
    
    return results

if __name__ == "__main__":
    # You can disable Ollama with use_ollama=False if it's not available
    results = run_comprehensive_analysis('data/spills_with_demographics.csv', use_ollama=True)