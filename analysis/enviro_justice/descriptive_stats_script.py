import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file):
    """Load and prepare data with all necessary variables"""
    
    print("LOADING AND PREPARING DATA")
    print("="*50)
    
    df = pd.read_csv(csv_file)
    
    # Convert dates
    df['Date of Discovery'] = pd.to_datetime(df['Date of Discovery'], errors='coerce')
    df['Initial Report Date'] = pd.to_datetime(df['Initial Report Date'], errors='coerce')
    
    # Calculate derived variables
    df['reporting_delay_days'] = (df['Initial Report Date'] - df['Date of Discovery']).dt.days
    df['discovery_year'] = df['Date of Discovery'].dt.year
    df['report_year'] = df['Initial Report Date'].dt.year
    
    # Volume calculations
    volume_columns = ['Oil BBLs Spilled', 'Condensate BBLs Spilled', 'Produced Water BBLs Spilled', 
                      'Drilling Fluid BBLs Spilled', 'Flow Back Fluid BBLs Spilled']
    df['total_volume_bbls'] = 0
    for col in volume_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['total_volume_bbls'] += df[col]
    
    # Binary indicators
    df['major_spill'] = (df['More than five barrels spilled'].astype(str) == 'Y')
    df['outside_berms'] = (df['Spilled outside of berms'].astype(str) == 'Y')
    df['historical_spill'] = (df['Spill Type'] == 'Historical')
    
    # Environmental impacts
    for impact in ['soil', 'groundwater', 'Surface Water']:
        if impact in df.columns:
            df[f'{impact}_impacted'] = (df[impact] == 1.0)
    
    # Demographic categories
    df['high_poverty'] = df['percent_poverty'] > 15
    df['minority_community'] = df['percent_white'] < 70
    df['low_income'] = df['median_household_income'] < df['median_household_income'].median()
    
    # Create demographic category labels
    df['poverty_category'] = df['high_poverty'].map({True: 'High Poverty (>15%)', False: 'Low Poverty (≤15%)'})
    df['race_category'] = df['minority_community'].map({True: 'Minority Community (<70% White)', False: 'Majority White (≥70% White)'})
    
    print(f"Data loaded: {len(df):,} records from {df['discovery_year'].min()}-{df['discovery_year'].max()}")
    
    return df

def dataset_overview_stats(df):
    """Generate overall dataset descriptive statistics"""
    
    print("\nDATASET OVERVIEW")
    print("="*50)
    
    overview = {
        'Metric': [
            'Total Spill Incidents',
            'Study Period',
            'Historical Spills',
            'Recent Spills',
            'Records with Volume Data',
            'Records with Geographic Data',
            'Records with Complete Demographic Data',
            'Unique Counties',
            'Unique Facility Types'
        ],
        'Value': [
            f"{len(df):,}",
            f"{df['discovery_year'].min():.0f}-{df['discovery_year'].max():.0f}",
            f"{(df['Spill Type'] == 'Historical').sum():,} ({(df['Spill Type'] == 'Historical').mean()*100:.1f}%)",
            f"{(df['Spill Type'] == 'Recent').sum():,} ({(df['Spill Type'] == 'Recent').mean()*100:.1f}%)",
            f"{(df['total_volume_bbls'] > 0).sum():,} ({(df['total_volume_bbls'] > 0).mean()*100:.1f}%)",
            f"{df[['Latitude', 'Longitude']].dropna().shape[0]:,} ({df[['Latitude', 'Longitude']].dropna().shape[0]/len(df)*100:.1f}%)",
            f"{df[['percent_poverty', 'percent_white', 'median_household_income']].dropna().shape[0]:,}",
            f"{df['county'].nunique():,}" if 'county' in df.columns else "N/A",
            f"{df['Facility Type'].nunique():,}" if 'Facility Type' in df.columns else "N/A"
        ]
    }
    
    overview_df = pd.DataFrame(overview)
    print(overview_df.to_string(index=False))
    
    return overview_df

def demographic_descriptive_stats(df):
    """Generate demographic descriptive statistics"""
    
    print("\nDEMOGRAFIC CHARACTERISTICS")
    print("="*50)
    
    # Overall demographic distribution
    demo_stats = {
        'Characteristic': [
            'Poverty Rate (%)',
            'Percent White (%)', 
            'Percent Hispanic (%)',
            'Median Household Income ($)',
            'Unemployment Rate (%)'
        ],
        'Mean': [
            f"{df['percent_poverty'].mean():.1f}",
            f"{df['percent_white'].mean():.1f}",
            f"{df['percent_hispanic'].mean():.1f}",
            f"${df['median_household_income'].mean():,.0f}",
            f"{df['unemployment_rate'].mean():.1f}" if 'unemployment_rate' in df.columns else "N/A"
        ],
        'Median': [
            f"{df['percent_poverty'].median():.1f}",
            f"{df['percent_white'].median():.1f}",
            f"{df['percent_hispanic'].median():.1f}",
            f"${df['median_household_income'].median():,.0f}",
            f"{df['unemployment_rate'].median():.1f}" if 'unemployment_rate' in df.columns else "N/A"
        ],
        'Std Dev': [
            f"{df['percent_poverty'].std():.1f}",
            f"{df['percent_white'].std():.1f}",
            f"{df['percent_hispanic'].std():.1f}",
            f"${df['median_household_income'].std():,.0f}",
            f"{df['unemployment_rate'].std():.1f}" if 'unemployment_rate' in df.columns else "N/A"
        ],
        'Range': [
            f"{df['percent_poverty'].min():.1f} - {df['percent_poverty'].max():.1f}",
            f"{df['percent_white'].min():.1f} - {df['percent_white'].max():.1f}",
            f"{df['percent_hispanic'].min():.1f} - {df['percent_hispanic'].max():.1f}",
            f"${df['median_household_income'].min():,.0f} - ${df['median_household_income'].max():,.0f}",
            f"{df['unemployment_rate'].min():.1f} - {df['unemployment_rate'].max():.1f}" if 'unemployment_rate' in df.columns else "N/A"
        ]
    }
    
    demo_df = pd.DataFrame(demo_stats)
    print(demo_df.to_string(index=False))
    
    # Community classifications
    print(f"\nCOMMUNITY CLASSIFICATIONS:")
    print(f"High Poverty Areas (>15%): {df['high_poverty'].sum():,} spills ({df['high_poverty'].mean()*100:.1f}%)")
    print(f"Minority Communities (<70% White): {df['minority_community'].sum():,} spills ({df['minority_community'].mean()*100:.1f}%)")
    print(f"Low-Income Areas (Below Median): {df['low_income'].sum():,} spills ({df['low_income'].mean()*100:.1f}%)")
    
    return demo_df

def spill_characteristics_by_demographics(df):
    """Generate spill characteristics by demographic groups"""
    
    print("\nSPILL CHARACTERISTICS BY DEMOGRAPHICS")
    print("="*50)
    
    # Create comparison table
    groups = ['Overall', 'High Poverty', 'Low Poverty', 'Minority Community', 'Majority White']
    characteristics = []
    
    for group in groups:
        if group == 'Overall':
            subset = df
        elif group == 'High Poverty':
            subset = df[df['high_poverty']]
        elif group == 'Low Poverty':
            subset = df[~df['high_poverty']]
        elif group == 'Minority Community':
            subset = df[df['minority_community']]
        else:  # Majority White
            subset = df[~df['minority_community']]
        
        characteristics.append({
            'Group': group,
            'N': f"{len(subset):,}",
            'Historical Spills (%)': f"{(subset['Spill Type'] == 'Historical').mean()*100:.1f}",
            'Major Spills (%)': f"{subset['major_spill'].mean()*100:.1f}",
            'Mean Volume (bbls)': f"{subset['total_volume_bbls'].mean():.2f}",
            'Median Volume (bbls)': f"{subset['total_volume_bbls'].median():.2f}",
            'Mean Reporting Delay (days)': f"{subset['reporting_delay_days'].mean():.1f}",
            'Median Reporting Delay (days)': f"{subset['reporting_delay_days'].median():.1f}",
            'Outside Berms (%)': f"{subset['outside_berms'].mean()*100:.1f}" if 'outside_berms' in subset.columns else "N/A"
        })
    
    char_df = pd.DataFrame(characteristics)
    print(char_df.to_string(index=False))
    
    return char_df

def environmental_impact_stats(df):
    """Generate environmental impact statistics"""
    
    print("\nENVIRONMENTAL IMPACT PATTERNS")
    print("="*50)
    
    impact_stats = []
    
    # Overall impact rates
    impact_types = ['soil_impacted', 'groundwater_impacted', 'Surface Water_impacted']
    impact_labels = ['Soil Contamination', 'Groundwater Contamination', 'Surface Water Contamination']
    
    for impact_type, label in zip(impact_types, impact_labels):
        if impact_type in df.columns:
            overall_rate = df[impact_type].mean() * 100
            hist_rate = df[df['historical_spill']][impact_type].mean() * 100
            recent_rate = df[~df['historical_spill']][impact_type].mean() * 100
            high_pov_rate = df[df['high_poverty']][impact_type].mean() * 100
            low_pov_rate = df[~df['high_poverty']][impact_type].mean() * 100
            
            impact_stats.append({
                'Impact Type': label,
                'Overall (%)': f"{overall_rate:.1f}",
                'Historical Spills (%)': f"{hist_rate:.1f}",
                'Recent Spills (%)': f"{recent_rate:.1f}",
                'High Poverty (%)': f"{high_pov_rate:.1f}",
                'Low Poverty (%)': f"{low_pov_rate:.1f}"
            })
    
    if impact_stats:
        impact_df = pd.DataFrame(impact_stats)
        print(impact_df.to_string(index=False))
    else:
        print("Environmental impact data not available")
        impact_df = pd.DataFrame()
    
    return impact_df

def temporal_patterns_stats(df):
    """Generate temporal pattern statistics"""
    
    print("\nTEMPORAL PATTERNS")
    print("="*50)
    
    # Filter to main study period
    df_temporal = df[(df['discovery_year'] >= 2014) & (df['discovery_year'] <= 2024)]
    
    # Annual summary statistics
    annual_stats = df_temporal.groupby('discovery_year').agg({
        'Spill Type': ['count', lambda x: (x == 'Historical').sum()],
        'major_spill': 'mean',
        'total_volume_bbls': ['mean', 'median'],
        'high_poverty': 'mean',
        'reporting_delay_days': 'mean'
    }).round(2)
    
    # Flatten column names
    annual_stats.columns = ['Total_Spills', 'Historical_Count', 'Major_Spill_Rate', 
                           'Mean_Volume', 'Median_Volume', 'High_Poverty_Rate', 'Mean_Delay']
    
    annual_stats['Historical_Rate'] = (annual_stats['Historical_Count'] / annual_stats['Total_Spills'] * 100).round(1)
    annual_stats['Major_Spill_Rate'] = (annual_stats['Major_Spill_Rate'] * 100).round(1)
    annual_stats['High_Poverty_Rate'] = (annual_stats['High_Poverty_Rate'] * 100).round(1)
    
    print("ANNUAL TRENDS (2014-2024):")
    print(annual_stats[['Total_Spills', 'Historical_Rate', 'Major_Spill_Rate', 'Mean_Volume', 'High_Poverty_Rate']].to_string())
    
    # Period comparison
    early_period = df_temporal[df_temporal['discovery_year'] <= 2018]
    recent_period = df_temporal[df_temporal['discovery_year'] >= 2020]
    
    print(f"\nPERIOD COMPARISON:")
    print(f"Early Period (2014-2018): {len(early_period):,} spills")
    print(f"Recent Period (2020-2024): {len(recent_period):,} spills")
    print(f"Historical Rate - Early: {(early_period['Spill Type'] == 'Historical').mean()*100:.1f}%")
    print(f"Historical Rate - Recent: {(recent_period['Spill Type'] == 'Historical').mean()*100:.1f}%")
    print(f"Major Spill Rate - Early: {early_period['major_spill'].mean()*100:.1f}%")
    print(f"Major Spill Rate - Recent: {recent_period['major_spill'].mean()*100:.1f}%")
    
    return annual_stats

def facility_type_stats(df):
    """Generate facility type statistics"""
    
    print("\nFACILITY TYPE DISTRIBUTION")
    print("="*50)
    
    if 'Facility Type' in df.columns:
        # Overall facility distribution
        facility_counts = df['Facility Type'].value_counts()
        facility_pct = (facility_counts / len(df) * 100).round(1)
        
        # Top 15 facility types
        top_facilities = pd.DataFrame({
            'Facility Type': facility_counts.head(15).index,
            'Count': facility_counts.head(15).values,
            'Percentage': facility_pct.head(15).values
        })
        
        print("TOP 15 FACILITY TYPES:")
        print(top_facilities.to_string(index=False))
        
        # Facility types by demographics
        print(f"\nFACILITY CONCENTRATION IN HIGH-POVERTY AREAS:")
        facility_demo = pd.crosstab(df['Facility Type'], df['high_poverty'], normalize='columns') * 100
        
        # Show facilities with highest concentration in high-poverty areas
        high_poverty_concentration = facility_demo[True].sort_values(ascending=False).head(10)
        for facility, pct in high_poverty_concentration.items():
            low_pov_pct = facility_demo.loc[facility, False] if facility in facility_demo.index else 0
            ratio = pct / low_pov_pct if low_pov_pct > 0 else float('inf')
            print(f"  {facility}: {pct:.1f}% (vs {low_pov_pct:.1f}% in low-poverty, {ratio:.1f}x ratio)")
        
        return top_facilities
    else:
        print("Facility type data not available")
        return pd.DataFrame()

def volume_distribution_stats(df):
    """Generate detailed volume distribution statistics"""
    
    print("\nSPILL VOLUME DISTRIBUTION")
    print("="*50)
    
    # Overall volume statistics
    volume_data = df[df['total_volume_bbls'] > 0]['total_volume_bbls']
    
    if len(volume_data) > 0:
        volume_stats = {
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', '25th Percentile', 
                         '75th Percentile', '90th Percentile', '95th Percentile', '99th Percentile', 'Max'],
            'Value (bbls)': [
                f"{len(volume_data):,}",
                f"{volume_data.mean():.2f}",
                f"{volume_data.median():.2f}",
                f"{volume_data.std():.2f}",
                f"{volume_data.min():.2f}",
                f"{volume_data.quantile(0.25):.2f}",
                f"{volume_data.quantile(0.75):.2f}",
                f"{volume_data.quantile(0.90):.2f}",
                f"{volume_data.quantile(0.95):.2f}",
                f"{volume_data.quantile(0.99):.2f}",
                f"{volume_data.max():.2f}"
            ]
        }
        
        volume_df = pd.DataFrame(volume_stats)
        print(volume_df.to_string(index=False))
        
        # Volume by categories
        print(f"\nVOLUME CATEGORIES:")
        print(f"Small spills (≤1 bbl): {(volume_data <= 1).sum():,} ({(volume_data <= 1).mean()*100:.1f}%)")
        print(f"Medium spills (1-5 bbls): {((volume_data > 1) & (volume_data <= 5)).sum():,} ({((volume_data > 1) & (volume_data <= 5)).mean()*100:.1f}%)")
        print(f"Large spills (5-50 bbls): {((volume_data > 5) & (volume_data <= 50)).sum():,} ({((volume_data > 5) & (volume_data <= 50)).mean()*100:.1f}%)")
        print(f"Very large spills (>50 bbls): {(volume_data > 50).sum():,} ({(volume_data > 50).mean()*100:.1f}%)")
        
        return volume_df
    else:
        print("No volume data available")
        return pd.DataFrame()

def save_all_descriptive_stats(df, filename='descriptive_statistics_tables.xlsx'):
    """Save all descriptive statistics to Excel file"""
    
    print(f"\nSAVING ALL TABLES TO {filename}")
    print("="*50)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Generate all tables
        overview = dataset_overview_stats(df)
        demographics = demographic_descriptive_stats(df)
        characteristics = spill_characteristics_by_demographics(df)
        impacts = environmental_impact_stats(df)
        temporal = temporal_patterns_stats(df)
        facilities = facility_type_stats(df)
        volumes = volume_distribution_stats(df)
        
        # Save to Excel
        overview.to_excel(writer, sheet_name='Dataset Overview', index=False)
        demographics.to_excel(writer, sheet_name='Demographics', index=False)
        characteristics.to_excel(writer, sheet_name='Spill Characteristics', index=False)
        if not impacts.empty:
            impacts.to_excel(writer, sheet_name='Environmental Impacts', index=False)
        temporal.to_excel(writer, sheet_name='Temporal Patterns', index=True)
        if not facilities.empty:
            facilities.to_excel(writer, sheet_name='Facility Types', index=False)
        if not volumes.empty:
            volumes.to_excel(writer, sheet_name='Volume Distribution', index=False)
    
    print(f"All descriptive statistics saved to {filename}")

def run_full_descriptive_analysis(csv_file):
    """Run complete descriptive statistics analysis"""
    
    print("COMPREHENSIVE DESCRIPTIVE STATISTICS ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data(csv_file)
    
    # Generate all descriptive statistics
    dataset_overview_stats(df)
    demographic_descriptive_stats(df)
    spill_characteristics_by_demographics(df)
    environmental_impact_stats(df)
    temporal_patterns_stats(df)
    facility_type_stats(df)
    volume_distribution_stats(df)
    
    # Save all tables
    save_all_descriptive_stats(df)
    
    print(f"\nDESCRIPTIVE ANALYSIS COMPLETE!")
    print(f"Tables saved to: descriptive_statistics_tables.xlsx")
    
    return df

if __name__ == "__main__":
    df = run_full_descriptive_analysis('data/spills_with_demographics.csv')