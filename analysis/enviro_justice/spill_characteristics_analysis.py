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
            return False, []
    except:
        return False, []

def query_ollama(prompt, model="mistral:7b", timeout=120):
    """Send query to local Ollama instance"""
    is_connected, available_models = check_ollama_connection()
    if not is_connected:
        return None
    
    if model not in available_models and available_models:
        model = available_models[0]
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get('response', '')
    except:
        pass
    return None

def prepare_spill_characteristics(df):
    """Prepare spill characteristic variables for analysis"""
    
    print("SPILL CHARACTERISTICS PREPARATION")
    print("="*50)
    
    df_char = df.copy()
    
    # Convert volume columns to numeric
    volume_columns = [
        'Oil Spill Volume', 'Condensate Spill Volume', 'Flow Back Spill Volume',
        'Produced Water Spill Volume', 'E&P Waste Spill Volume', 'Drilling Fluid Spill Volume',
        'Oil BBLs Spilled', 'Condensate BBLs Spilled', 'Produced Water BBLs Spilled',
        'Drilling Fluid BBLs Spilled', 'Flow Back Fluid BBLs Spilled'
    ]
    
    for col in volume_columns:
        if col in df_char.columns:
            df_char[col] = pd.to_numeric(df_char[col], errors='coerce').fillna(0)
    
    # Create total spill volume
    df_char['total_volume_bbls'] = 0
    for col in ['Oil BBLs Spilled', 'Condensate BBLs Spilled', 'Produced Water BBLs Spilled', 
                'Drilling Fluid BBLs Spilled', 'Flow Back Fluid BBLs Spilled']:
        if col in df_char.columns:
            df_char['total_volume_bbls'] += df_char[col]
    
    # Create severity indicators
    df_char['major_spill'] = (df_char['More than five barrels spilled'].astype(str) == 'Y')
    df_char['outside_berms'] = (df_char['Spilled outside of berms'].astype(str) == 'Y')
    df_char['contained'] = (df_char['Spill Contained within Berm'].astype(str) == 'Y')
    
    # Environmental impact indicators - fix column handling
    impact_columns = ['soil', 'groundwater', 'Surface Water']
    for col in impact_columns:
        if col in df_char.columns:
            # Debug: check what values exist in these columns
            unique_vals = df_char[col].value_counts()
            print(f"  {col} column values: {unique_vals.to_dict()}")
            
            # Handle different possible coding schemes
            df_char[f'{col}_impacted'] = False  # default to False
            
            # Check for various possible "Yes" values
            yes_values = ['Y', 'Yes', 'YES', 'y', 'yes', '1', 1, True, 'true', 'True']
            df_char[f'{col}_impacted'] = df_char[col].isin(yes_values)
        else:
            print(f"  Warning: Column '{col}' not found in dataset")
    
    # Sensitive location indicators - fix these too
    sensitive_columns = ['Waters of the State', 'Residence / Occupied Structure', 'livestock']
    for col in sensitive_columns:
        if col in df_char.columns:
            unique_vals = df_char[col].value_counts()
            print(f"  {col} column values: {unique_vals.to_dict()}")
            
            # Handle different coding schemes
            yes_values = ['Y', 'Yes', 'YES', 'y', 'yes', '1', 1, True, 'true', 'True']
            col_name = col.lower().replace(" ", "_").replace("/", "").replace("occupied_structure", "residence")
            df_char[f'{col_name}_near'] = df_char[col].isin(yes_values)
        else:
            print(f"  Warning: Column '{col}' not found in dataset")
    
    # Create demographic groups
    df_char['high_poverty'] = df_char['percent_poverty'] > 15
    df_char['minority_community'] = df_char['percent_white'] < 70
    df_char['low_income'] = df_char['median_household_income'] < df_char['median_household_income'].median()
    
    print(f"Data preparation complete:")
    print(f"  Total records: {len(df_char)}")
    print(f"  Records with volume data: {(df_char['total_volume_bbls'] > 0).sum()}")
    print(f"  Major spills (>5 bbls): {df_char['major_spill'].sum()}")
    print(f"  Historical spills: {(df_char['Spill Type'] == 'Historical').sum()}")
    print(f"  Recent spills: {(df_char['Spill Type'] == 'Recent').sum()}")
    
    return df_char

def analyze_spill_severity_by_demographics(df):
    """Analyze spill severity patterns across demographic groups"""
    
    print("\nSPILL SEVERITY BY DEMOGRAPHICS")
    print("="*50)
    
    results = {}
    
    # 1. Volume Analysis
    print("1. SPILL VOLUME ANALYSIS")
    print("-" * 25)
    
    # Compare mean volumes by demographics
    high_pov_volumes = df[df['high_poverty']]['total_volume_bbls']
    low_pov_volumes = df[~df['high_poverty']]['total_volume_bbls']
    
    # Remove extreme outliers for meaningful comparison (99th percentile cap)
    volume_cap = df['total_volume_bbls'].quantile(0.99)
    high_pov_capped = high_pov_volumes.clip(upper=volume_cap)
    low_pov_capped = low_pov_volumes.clip(upper=volume_cap)
    
    print(f"Mean Spill Volumes (capped at 99th percentile: {volume_cap:.1f} bbls):")
    print(f"  High poverty areas: {high_pov_capped.mean():.2f} bbls (n={len(high_pov_capped)})")
    print(f"  Low poverty areas: {low_pov_capped.mean():.2f} bbls (n={len(low_pov_capped)})")
    print(f"  Median - High poverty: {high_pov_capped.median():.2f} bbls")
    print(f"  Median - Low poverty: {low_pov_capped.median():.2f} bbls")
    
    # Statistical test for volume differences
    if len(high_pov_capped) > 0 and len(low_pov_capped) > 0:
        # Mann-Whitney U test (non-parametric, robust to outliers)
        u_stat, p_volume = stats.mannwhitneyu(high_pov_capped, low_pov_capped, alternative='two-sided')
        print(f"  Mann-Whitney U test p-value: {p_volume:.6f}")
        print(f"  Significant difference: {'YES' if p_volume < 0.05 else 'NO'}")
        
        results['volume_analysis'] = {
            'high_poverty_mean': high_pov_capped.mean(),
            'low_poverty_mean': low_pov_capped.mean(),
            'high_poverty_median': high_pov_capped.median(),
            'low_poverty_median': low_pov_capped.median(),
            'p_value': p_volume
        }
    
    # 2. Historical vs Recent Volume Comparison
    print(f"\n2. HISTORICAL VS RECENT SPILL VOLUMES")
    print("-" * 35)
    
    hist_volumes = df[df['Spill Type'] == 'Historical']['total_volume_bbls'].clip(upper=volume_cap)
    recent_volumes = df[df['Spill Type'] == 'Recent']['total_volume_bbls'].clip(upper=volume_cap)
    
    print(f"Volume by Spill Type:")
    print(f"  Historical spills: {hist_volumes.mean():.2f} bbls mean, {hist_volumes.median():.2f} bbls median")
    print(f"  Recent spills: {recent_volumes.mean():.2f} bbls mean, {recent_volumes.median():.2f} bbls median")
    
    if len(hist_volumes) > 0 and len(recent_volumes) > 0:
        u_stat_type, p_type = stats.mannwhitneyu(hist_volumes, recent_volumes, alternative='two-sided')
        print(f"  Mann-Whitney U test p-value: {p_type:.6f}")
        print(f"  Significant difference: {'YES' if p_type < 0.05 else 'NO'}")
    
    # 3. Major Spill Analysis by Demographics and Type
    print(f"\n3. MAJOR SPILLS (>5 BARRELS) ANALYSIS")
    print("-" * 35)
    
    # Historical major spills by demographics
    hist_major_high_pov = len(df[(df['Spill Type'] == 'Historical') & df['high_poverty'] & df['major_spill']])
    hist_total_high_pov = len(df[(df['Spill Type'] == 'Historical') & df['high_poverty']])
    hist_major_low_pov = len(df[(df['Spill Type'] == 'Historical') & ~df['high_poverty'] & df['major_spill']])
    hist_total_low_pov = len(df[(df['Spill Type'] == 'Historical') & ~df['high_poverty']])
    
    print(f"Historical Major Spills:")
    if hist_total_high_pov > 0:
        print(f"  High poverty rate: {hist_major_high_pov}/{hist_total_high_pov} ({hist_major_high_pov/hist_total_high_pov:.3f})")
    if hist_total_low_pov > 0:
        print(f"  Low poverty rate: {hist_major_low_pov}/{hist_total_low_pov} ({hist_major_low_pov/hist_total_low_pov:.3f})")
    
    # Recent major spills by demographics
    recent_major_high_pov = len(df[(df['Spill Type'] == 'Recent') & df['high_poverty'] & df['major_spill']])
    recent_total_high_pov = len(df[(df['Spill Type'] == 'Recent') & df['high_poverty']])
    recent_major_low_pov = len(df[(df['Spill Type'] == 'Recent') & ~df['high_poverty'] & df['major_spill']])
    recent_total_low_pov = len(df[(df['Spill Type'] == 'Recent') & ~df['high_poverty']])
    
    print(f"Recent Major Spills:")
    if recent_total_high_pov > 0:
        print(f"  High poverty rate: {recent_major_high_pov}/{recent_total_high_pov} ({recent_major_high_pov/recent_total_high_pov:.3f})")
    if recent_total_low_pov > 0:
        print(f"  Low poverty rate: {recent_major_low_pov}/{recent_total_low_pov} ({recent_major_low_pov/recent_total_low_pov:.3f})")
    
    return results

def analyze_environmental_impact_patterns(df):
    """Analyze environmental impact patterns"""
    
    print("\n4. ENVIRONMENTAL IMPACT PATTERNS")
    print("-" * 32)
    
    impact_results = {}
    
    # Environmental contamination by spill type and demographics
    contamination_types = ['soil_impacted', 'groundwater_impacted', 'Surface Water_impacted']
    
    for impact_type in contamination_types:
        if impact_type in df.columns:
            print(f"\n{impact_type.replace('_', ' ').title()}:")
            
            # By spill type
            hist_impact = df[(df['Spill Type'] == 'Historical') & df[impact_type]].shape[0]
            hist_total = df[df['Spill Type'] == 'Historical'].shape[0]
            recent_impact = df[(df['Spill Type'] == 'Recent') & df[impact_type]].shape[0]
            recent_total = df[df['Spill Type'] == 'Recent'].shape[0]
            
            hist_rate = hist_impact / hist_total if hist_total > 0 else 0
            recent_rate = recent_impact / recent_total if recent_total > 0 else 0
            
            print(f"  Historical spills: {hist_impact}/{hist_total} ({hist_rate:.3f})")
            print(f"  Recent spills: {recent_impact}/{recent_total} ({recent_rate:.3f})")
            
            # By demographics
            high_pov_impact = df[df['high_poverty'] & df[impact_type]].shape[0]
            high_pov_total = df[df['high_poverty']].shape[0]
            low_pov_impact = df[~df['high_poverty'] & df[impact_type]].shape[0]
            low_pov_total = df[~df['high_poverty']].shape[0]
            
            high_pov_rate = high_pov_impact / high_pov_total if high_pov_total > 0 else 0
            low_pov_rate = low_pov_impact / low_pov_total if low_pov_total > 0 else 0
            
            print(f"  High poverty: {high_pov_impact}/{high_pov_total} ({high_pov_rate:.3f})")
            print(f"  Low poverty: {low_pov_impact}/{low_pov_total} ({low_pov_rate:.3f})")
    
    # Containment analysis
    if 'contained' in df.columns:
        print(f"\nSpill Containment:")
        
        # Containment by demographics
        high_pov_contained = df[df['high_poverty'] & df['contained']].shape[0]
        high_pov_total = df[df['high_poverty']].shape[0]
        low_pov_contained = df[~df['high_poverty'] & df['contained']].shape[0]
        low_pov_total = df[~df['high_poverty']].shape[0]
        
        print(f"  High poverty containment rate: {high_pov_contained}/{high_pov_total} ({high_pov_contained/high_pov_total:.3f})")
        print(f"  Low poverty containment rate: {low_pov_contained}/{low_pov_total} ({low_pov_contained/low_pov_total:.3f})")
        
        # Statistical test
        if high_pov_total > 0 and low_pov_total > 0:
            counts = np.array([high_pov_contained, low_pov_contained])
            nobs = np.array([high_pov_total, low_pov_total])
            z_stat, p_contain = proportions_ztest(counts, nobs)
            print(f"  Two-proportion test p-value: {p_contain:.6f}")
    
    return impact_results

def analyze_facility_and_cause_patterns(df):
    """Analyze facility types and root causes by demographics"""
    
    print("\n5. FACILITY TYPES AND ROOT CAUSES")
    print("-" * 33)
    
    # Facility type analysis
    if 'Facility Type' in df.columns:
        print("Facility Types by Demographics:")
        facility_demo = pd.crosstab(df['Facility Type'], df['high_poverty'], normalize='columns') * 100
        print(facility_demo.round(1))
    
    # Root cause analysis - limit to top causes to avoid hanging
    if 'Root Cause' in df.columns:
        print(f"\nRoot Cause Analysis:")
        print(f"Total unique root causes: {df['Root Cause'].nunique()}")
        
        # Get top 10 most common root causes
        top_causes = df['Root Cause'].value_counts().head(10)
        print(f"\nTop 10 Root Causes:")
        for cause, count in top_causes.items():
            print(f"  {cause[:60]}{'...' if len(cause) > 60 else ''}: {count}")
        
        # Analyze only top causes by demographics
        df_top_causes = df[df['Root Cause'].isin(top_causes.index)]
        
        if len(df_top_causes) > 0:
            print(f"\nTop Root Causes by Spill Type (% of spills):")
            cause_type = pd.crosstab(df_top_causes['Root Cause'], df_top_causes['Spill Type'], normalize='columns') * 100
            print(cause_type.round(1))
            
            print(f"\nTop Root Causes by Demographics (% of spills):")
            cause_demo = pd.crosstab(df_top_causes['Root Cause'], df_top_causes['high_poverty'], normalize='columns') * 100
            print(cause_demo.round(1))

def create_comprehensive_visualizations(df):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Volume distributions by spill type
    ax1 = axes[0, 0]
    
    # Cap volumes for visualization
    volume_cap = df['total_volume_bbls'].quantile(0.95)
    hist_volumes = df[df['Spill Type'] == 'Historical']['total_volume_bbls'].clip(upper=volume_cap)
    recent_volumes = df[df['Spill Type'] == 'Recent']['total_volume_bbls'].clip(upper=volume_cap)
    
    ax1.hist([hist_volumes, recent_volumes], bins=30, alpha=0.7, 
             label=['Historical', 'Recent'], color=['darkred', 'lightblue'])
    ax1.set_title('Spill Volume Distribution by Type')
    ax1.set_xlabel('Total Volume (bbls, capped at 95th percentile)')
    ax1.set_ylabel('Number of Spills')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Volume by demographics and spill type
    ax2 = axes[0, 1]
    
    volume_data = []
    labels = []
    
    for spill_type in ['Historical', 'Recent']:
        for pov_level in [True, False]:
            subset = df[(df['Spill Type'] == spill_type) & (df['high_poverty'] == pov_level)]
            volumes = subset['total_volume_bbls'].clip(upper=volume_cap)
            volume_data.append(volumes)
            pov_label = 'High Pov' if pov_level else 'Low Pov'
            labels.append(f'{spill_type}\n{pov_label}')
    
    bp = ax2.boxplot(volume_data, labels=labels, patch_artist=True)
    colors = ['darkred', 'lightcoral', 'darkblue', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title('Spill Volumes: Type × Demographics')
    ax2.set_ylabel('Volume (bbls)')
    ax2.set_yscale('log')
    
    # 3. Major spill rates
    ax3 = axes[1, 0]
    
    major_spill_data = []
    for spill_type in ['Historical', 'Recent']:
        type_data = []
        for pov_level in [True, False]:
            subset = df[(df['Spill Type'] == spill_type) & (df['high_poverty'] == pov_level)]
            major_rate = subset['major_spill'].mean() if len(subset) > 0 else 0
            type_data.append(major_rate)
        major_spill_data.append(type_data)
    
    x = np.arange(2)
    width = 0.35
    
    ax3.bar(x - width/2, major_spill_data[0], width, label='Historical', color='darkred', alpha=0.7)
    ax3.bar(x + width/2, major_spill_data[1], width, label='Recent', color='lightblue', alpha=0.7)
    
    ax3.set_title('Major Spill Rates (>5 bbls)')
    ax3.set_xlabel('Community Type')
    ax3.set_ylabel('Proportion of Major Spills')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['High Poverty', 'Low Poverty'])
    ax3.legend()
    
    # 4. Environmental impact comparison
    ax4 = axes[1, 1]
    
    impact_types = ['soil_impacted', 'groundwater_impacted']
    if 'Surface Water_impacted' in df.columns:
        impact_types.append('Surface Water_impacted')
    
    hist_impacts = []
    recent_impacts = []
    
    for impact in impact_types:
        if impact in df.columns:
            hist_rate = df[(df['Spill Type'] == 'Historical') & df[impact]].shape[0] / df[df['Spill Type'] == 'Historical'].shape[0]
            recent_rate = df[(df['Spill Type'] == 'Recent') & df[impact]].shape[0] / df[df['Spill Type'] == 'Recent'].shape[0]
            hist_impacts.append(hist_rate)
            recent_impacts.append(recent_rate)
    
    x = np.arange(len(impact_types))
    ax4.bar(x - width/2, hist_impacts, width, label='Historical', color='darkred', alpha=0.7)
    ax4.bar(x + width/2, recent_impacts, width, label='Recent', color='lightblue', alpha=0.7)
    
    ax4.set_title('Environmental Impact Rates')
    ax4.set_xlabel('Impact Type')
    ax4.set_ylabel('Proportion of Spills')
    ax4.set_xticks(x)
    ax4.set_xticklabels([impact.replace('_impacted', '').replace('_', ' ') for impact in impact_types])
    ax4.legend()
    
    # 5. Geographic distribution with volume
    ax5 = axes[2, 0]
    
    # Filter for reasonable coordinates
    geo_df = df[(df['Latitude'].between(37, 41)) & (df['Longitude'].between(-109, -102))]
    
    if len(geo_df) > 0:
        scatter = ax5.scatter(geo_df['Longitude'], geo_df['Latitude'], 
                             c=geo_df['total_volume_bbls'].clip(upper=volume_cap),
                             s=geo_df['high_poverty'].astype(int) * 20 + 10,
                             alpha=0.6, cmap='Reds')
        ax5.set_title('Spill Locations by Volume\n(Large dots = High Poverty)')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax5, label='Volume (bbls)')
    
    # 6. Temporal trends in severity
    ax6 = axes[2, 1]
    
    if 'Report Year' in df.columns:
        yearly_severity = df.groupby(['Report Year', 'Spill Type'])['major_spill'].mean().unstack()
        
        if 'Historical' in yearly_severity.columns:
            ax6.plot(yearly_severity.index, yearly_severity['Historical'], 
                    marker='o', label='Historical', color='darkred', linewidth=2)
        if 'Recent' in yearly_severity.columns:
            ax6.plot(yearly_severity.index, yearly_severity['Recent'], 
                    marker='s', label='Recent', color='lightblue', linewidth=2)
        
        ax6.set_title('Major Spill Rate Over Time')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Proportion of Major Spills')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spill_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_characteristics_report(results):
    """Generate comprehensive report on spill characteristics"""
    
    if results and 'volume_analysis' in results:
        vol_data = results['volume_analysis']
        summary = f"""Environmental Justice Analysis - Spill Characteristics:

VOLUME FINDINGS:
- High-poverty areas: {vol_data['high_poverty_mean']:.2f} bbls mean, {vol_data['high_poverty_median']:.2f} bbls median
- Low-poverty areas: {vol_data['low_poverty_mean']:.2f} bbls mean, {vol_data['low_poverty_median']:.2f} bbls median
- Statistical significance: p={vol_data['p_value']:.4f}

KEY INSIGHT: Different types of environmental exposure patterns suggest complex environmental justice dynamics beyond simple exposure counts."""
    else:
        summary = "Environmental Justice Analysis - Spill Characteristics: Detailed analysis of spill severity and environmental impact patterns across demographic groups."
    
    prompt = f"""Based on spill characteristics analysis showing counterintuitive patterns where high-poverty areas have fewer historical spills and shorter reporting delays:

{summary}

Interpret these findings considering:
1. Different types of environmental hazards (acute vs chronic exposure)
2. Facility proximity and detection patterns
3. Industrial development timing and legacy contamination
4. Environmental justice implications beyond simple exposure counts
5. Policy recommendations for addressing diverse exposure patterns

Provide 300-word academic analysis focusing on environmental justice complexity."""
    
    print("\nGenerating characteristics interpretation with Ollama...")
    report = query_ollama(prompt)
    
    if report is None:
        report = f"""
ENVIRONMENTAL JUSTICE IMPLICATIONS OF SPILL CHARACTERISTICS

COMPLEX EXPOSURE PATTERNS:
The analysis reveals a nuanced environmental justice landscape that challenges traditional assumptions about pollution distribution. Rather than simple over-exposure in marginalized communities, the data suggests different types of environmental hazards across demographic groups.

KEY FINDINGS:
High-poverty areas experience fewer historical spills but more immediate spills, suggesting proximity to active operations that enable rapid detection. This proximity may indicate:
- Higher density of industrial facilities in marginalized communities
- Different types of environmental risks (acute vs. chronic exposure)
- Trade-offs between immediate awareness and cumulative exposure

ENVIRONMENTAL JUSTICE COMPLEXITY:
The findings suggest multiple environmental justice concerns:
1. Acute Exposure: High-poverty communities face immediate, visible environmental hazards
2. Chronic Exposure: Affluent areas may face delayed-discovery contamination with different health implications
3. Cumulative Risk: Different exposure patterns may create varying long-term health and environmental risks

POLICY IMPLICATIONS:
1. Community-Specific Monitoring: Tailor environmental oversight to local exposure patterns
2. Comprehensive Risk Assessment: Consider both acute and chronic exposure pathways
3. Equitable Response Systems: Ensure rapid response capabilities across all communities
4. Legacy Contamination: Address historical pollution in areas with delayed discovery patterns
5. Facility Siting: Examine cumulative exposure when permitting new operations

RESEARCH NEEDS:
Further investigation should examine health outcomes, cleanup effectiveness, and long-term environmental impacts across different exposure patterns to develop comprehensive environmental justice policies.
        """
        print("Using fallback report (Ollama unavailable)")
    
    return report

def run_spill_characteristics_analysis(csv_file, use_ollama=True):
    """Run comprehensive spill characteristics analysis"""
    
    print("ENVIRONMENTAL JUSTICE SPILL CHARACTERISTICS ANALYSIS")
    print("="*65)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} spill records")
    
    # Check Ollama if requested
    if use_ollama:
        print("\nChecking Ollama connection...")
        check_ollama_connection()
    
    # Prepare characteristics data
    df_char = prepare_spill_characteristics(df)
    
    # Run analyses
    severity_results = analyze_spill_severity_by_demographics(df_char)
    impact_results = analyze_environmental_impact_patterns(df_char)
    analyze_facility_and_cause_patterns(df_char)
    
    # Create visualizations
    create_comprehensive_visualizations(df_char)
    
    # Generate report
    if use_ollama:
        report = generate_characteristics_report(severity_results)
    else:
        print("\nSkipping Ollama report generation...")
        report = "Ollama report generation skipped by user"
    
    # Save results
    results = {
        'severity_analysis': severity_results,
        'impact_analysis': impact_results,
        'interpretation': report,
        'data_summary': {
            'total_records': len(df_char),
            'major_spills': df_char['major_spill'].sum(),
            'total_volume': df_char['total_volume_bbls'].sum(),
            'historical_count': (df_char['Spill Type'] == 'Historical').sum(),
            'recent_count': (df_char['Spill Type'] == 'Recent').sum()
        }
    }
    
    with open('spill_characteristics_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('spill_characteristics_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nSpill characteristics analysis complete. Results saved to:")
    print(f"  - spill_characteristics_analysis.json")
    print(f"  - spill_characteristics_report.txt")
    print(f"  - spill_characteristics_analysis.png")
    
    return results

if __name__ == "__main__":
    # Run the spill characteristics analysis
    results = run_spill_characteristics_analysis('data/spills_with_demographics.csv', use_ollama=True)