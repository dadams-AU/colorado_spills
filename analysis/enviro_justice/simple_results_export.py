import pandas as pd
import numpy as np
import json

def create_summary_results(csv_file):
    """Create summary results without hanging visualizations"""
    
    print("CREATING ENVIRONMENTAL JUSTICE SUMMARY")
    print("="*50)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Create basic variables
    df['high_poverty'] = df['percent_poverty'] > 15
    df['minority_community'] = df['percent_white'] < 70
    df['major_spill'] = (df['More than five barrels spilled'].astype(str) == 'Y')
    
    # Convert dates
    df['Date of Discovery'] = pd.to_datetime(df['Date of Discovery'], errors='coerce')
    df['Initial Report Date'] = pd.to_datetime(df['Initial Report Date'], errors='coerce')
    df['reporting_delay_days'] = (df['Initial Report Date'] - df['Date of Discovery']).dt.days
    
    # Volume analysis
    volume_columns = ['Oil BBLs Spilled', 'Condensate BBLs Spilled', 'Produced Water BBLs Spilled', 
                      'Drilling Fluid BBLs Spilled', 'Flow Back Fluid BBLs Spilled']
    df['total_volume_bbls'] = 0
    for col in volume_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['total_volume_bbls'] += df[col]
    
    # Environmental impact
    for col in ['soil', 'groundwater', 'Surface Water']:
        if col in df.columns:
            df[f'{col}_impacted'] = df[col] == 1.0
    
    # Create summary dictionary
    summary = {
        'dataset_overview': {
            'total_spills': len(df),
            'date_range': f"{df['Date of Discovery'].min().year}-{df['Date of Discovery'].max().year}",
            'historical_spills': (df['Spill Type'] == 'Historical').sum(),
            'recent_spills': (df['Spill Type'] == 'Recent').sum()
        },
        
        'demographic_patterns': {
            'high_poverty_spills': df['high_poverty'].sum(),
            'low_poverty_spills': (~df['high_poverty']).sum(),
            'minority_community_spills': df['minority_community'].sum(),
            'majority_white_spills': (~df['minority_community']).sum()
        },
        
        'spill_severity': {
            'high_poverty_mean_volume': df[df['high_poverty']]['total_volume_bbls'].mean(),
            'low_poverty_mean_volume': df[~df['high_poverty']]['total_volume_bbls'].mean(),
            'historical_mean_volume': df[df['Spill Type'] == 'Historical']['total_volume_bbls'].mean(),
            'recent_mean_volume': df[df['Spill Type'] == 'Recent']['total_volume_bbls'].mean(),
            'high_poverty_major_rate': df[df['high_poverty']]['major_spill'].mean(),
            'low_poverty_major_rate': df[~df['high_poverty']]['major_spill'].mean()
        },
        
        'reporting_delays': {
            'high_poverty_delay_mean': df[df['high_poverty']]['reporting_delay_days'].mean(),
            'low_poverty_delay_mean': df[~df['high_poverty']]['reporting_delay_days'].mean(),
            'historical_rate_high_poverty': (df['high_poverty'] & (df['Spill Type'] == 'Historical')).sum() / df['high_poverty'].sum(),
            'historical_rate_low_poverty': (~df['high_poverty'] & (df['Spill Type'] == 'Historical')).sum() / (~df['high_poverty']).sum()
        },
        
        'environmental_impacts': {},
        
        'facility_patterns': {}
    }
    
    # Environmental impacts
    for impact in ['soil', 'groundwater', 'Surface Water']:
        if f'{impact}_impacted' in df.columns:
            summary['environmental_impacts'][impact] = {
                'high_poverty_rate': df[df['high_poverty']][f'{impact}_impacted'].mean(),
                'low_poverty_rate': df[~df['high_poverty']][f'{impact}_impacted'].mean(),
                'historical_rate': df[df['Spill Type'] == 'Historical'][f'{impact}_impacted'].mean(),
                'recent_rate': df[df['Spill Type'] == 'Recent'][f'{impact}_impacted'].mean()
            }
    
    # Facility patterns
    if 'Facility Type' in df.columns:
        facility_counts = pd.crosstab(df['Facility Type'], df['high_poverty'], normalize='columns')
        summary['facility_patterns'] = facility_counts.to_dict()
    
    # Root causes (top 5 only)
    if 'Root Cause' in df.columns:
        top_causes = df['Root Cause'].value_counts().head(5)
        summary['top_root_causes'] = top_causes.to_dict()
    
    # Print key findings
    print(f"KEY FINDINGS SUMMARY:")
    print(f"  Total spills: {summary['dataset_overview']['total_spills']:,}")
    print(f"  High-poverty area spills: {summary['demographic_patterns']['high_poverty_spills']:,}")
    print(f"  Mean volume in high-poverty areas: {summary['spill_severity']['high_poverty_mean_volume']:.2f} bbls")
    print(f"  Mean volume in low-poverty areas: {summary['spill_severity']['low_poverty_mean_volume']:.2f} bbls")
    print(f"  Volume ratio (high/low poverty): {summary['spill_severity']['high_poverty_mean_volume']/summary['spill_severity']['low_poverty_mean_volume']:.2f}x")
    
    # Save results
    with open('environmental_justice_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: environmental_justice_summary.json")
    
    return summary

if __name__ == "__main__":
    results = create_summary_results('data/spills_with_demographics.csv')