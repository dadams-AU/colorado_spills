import pandas as pd
import requests
import json
from collections import Counter, defaultdict
import numpy as np

def query_ollama(prompt, model="mistral"):
    """Send query to local Ollama instance"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            })
        return response.json()['response']
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def analyze_spill_demographics(df):
    """Analyze demographic patterns in spill data"""
    
    # Basic demographic statistics
    demo_stats = {
        'total_spills': len(df),
        'avg_median_income': df['median_household_income'].mean(),
        'avg_poverty_rate': df['percent_poverty'].mean(),
        'avg_white_percentage': df['percent_white'].mean(),
        'avg_hispanic_percentage': df['percent_hispanic'].mean(),
        'avg_unemployment': df['unemployment_rate'].mean()
    }
    
    # Environmental justice analysis
    # Define high-poverty communities (>15% poverty rate)
    high_poverty = df[df['percent_poverty'] > 15]
    low_poverty = df[df['percent_poverty'] <= 15]
    
    # Define minority communities (>30% non-white)
    minority_communities = df[df['percent_white'] < 70]
    white_communities = df[df['percent_white'] >= 70]
    
    # Convert spill volumes to numeric, handling 'Unknown' values
    produced_water_numeric = pd.to_numeric(df['Produced Water Spill Volume'], errors='coerce')
    high_poverty_volumes = pd.to_numeric(high_poverty['Produced Water Spill Volume'], errors='coerce')
    
    ej_analysis = {
        'high_poverty_spills': len(high_poverty),
        'high_poverty_avg_volume': high_poverty_volumes.sum(),
        'minority_community_spills': len(minority_communities),
        'spills_by_income_quartile': df.groupby(pd.qcut(df['median_household_income'], 4, labels=['Q1(Lowest)', 'Q2', 'Q3', 'Q4(Highest)'])).size().to_dict(),
        'major_spills_by_poverty': {
            'high_poverty_major': len(high_poverty[high_poverty['More than five barrels spilled'] == 'Y']),
            'low_poverty_major': len(low_poverty[low_poverty['More than five barrels spilled'] == 'Y'])
        }
    }
    
    return demo_stats, ej_analysis

def analyze_root_causes(df):
    """Analyze already-categorized root causes"""
    
    # Count existing cause categories, handling NaN values
    cause_counts = {
        'human_error': df['Human Error'].fillna(0).sum(),
        'equipment_failure': df['Equipment Failure'].fillna(0).sum(), 
        'historical_unknown': df['Historical Unkown'].fillna(0).sum(),  # Note: typo in original data
        'other': df['Other'].fillna(0).sum()
    }
    
    # Get specific root cause descriptions
    root_causes = df['Root Cause'].dropna().value_counts().head(10)
    
    return cause_counts, root_causes

def analyze_spill_themes_llm(df, sample_size=50):
    """Use LLM to analyze themes in spill descriptions"""
    
    # Sample descriptions for LLM analysis (to avoid overwhelming it)
    descriptions_series = df['Spill Description'].dropna()
    if len(descriptions_series) == 0:
        return "No spill descriptions available for analysis."
    
    sample_descriptions = descriptions_series.sample(min(sample_size, len(descriptions_series))).tolist()
    
    # Combine descriptions for batch analysis
    combined_text = "\n---\n".join(sample_descriptions)
    
    prompt = f"""
    Analyze these oil and gas spill incident descriptions to identify themes and patterns.
    Focus on:
    1. Common equipment failures (tanks, valves, pipelines, etc.)
    2. Operational issues (overflow, leaks, maintenance problems)
    3. Environmental factors (weather, terrain, wildlife)
    4. Human factors (operator error, maintenance issues)
    5. Discovery methods (routine inspection, alarms, third-party reports)
    6. Spill severity indicators
    
    Incident descriptions:
    {combined_text}
    
    Provide a structured analysis with:
    - Top 5 equipment failure patterns
    - Most common operational issues  
    - Environmental risk factors
    - Human factor patterns
    - Recommendations for prevention based on these patterns
    
    Format as a concise regulatory summary suitable for policy recommendations.
    """
    
    return query_ollama(prompt)

def demographic_spill_analysis(df):
    """Analyze spill patterns by demographic characteristics"""
    
    # Create demographic categories
    df_analysis = df.copy()
    df_analysis['income_category'] = pd.cut(df_analysis['median_household_income'], 
                                          bins=3, labels=['Low Income', 'Middle Income', 'High Income'])
    df_analysis['poverty_category'] = pd.cut(df_analysis['percent_poverty'], 
                                           bins=[0, 10, 20, 100], labels=['Low Poverty', 'Moderate Poverty', 'High Poverty'])
    df_analysis['race_category'] = df_analysis['percent_white'].apply(
        lambda x: 'Majority White' if x >= 70 else 'Minority Community'
    )
    
    # Analyze spill patterns by demographics
    demo_patterns = {
        'spills_by_income': df_analysis.groupby('income_category').size().to_dict(),
        'spills_by_poverty': df_analysis.groupby('poverty_category').size().to_dict(),
        'spills_by_race': df_analysis.groupby('race_category').size().to_dict(),
        'volume_by_demographics': {
            'high_poverty_major_spills': len(df_analysis[(df_analysis['percent_poverty'] > 15) & 
                                                       (df_analysis['More than five barrels spilled'].astype(str) == 'Y')]),
            'minority_major_spills': len(df_analysis[(df_analysis['percent_white'] < 70) & 
                                                   (df_analysis['More than five barrels spilled'].astype(str) == 'Y')])
        }
    }
    
    return demo_patterns

def analyze_environmental_justice(df, sample_descriptions=20):
    """Use LLM to analyze environmental justice implications"""
    
    # Get descriptions from high-poverty and minority communities
    high_poverty_desc = df[df['percent_poverty'] > 15]['Spill Description'].dropna()
    minority_desc = df[df['percent_white'] < 70]['Spill Description'].dropna()
    
    if len(high_poverty_desc) == 0 or len(minority_desc) == 0:
        return "Insufficient data for environmental justice analysis."
    
    high_poverty_spills = high_poverty_desc.sample(min(sample_descriptions//2, len(high_poverty_desc))).tolist()
    minority_spills = minority_desc.sample(min(sample_descriptions//2, len(minority_desc))).tolist()
    
    combined_ej_text = "\n---HIGH POVERTY AREA---\n".join(high_poverty_spills) + "\n---MINORITY COMMUNITY---\n".join(minority_spills)
    
    prompt = f"""
    Analyze these spill incidents from high-poverty and minority communities for environmental justice concerns.
    
    Consider:
    1. Severity of incidents in vulnerable communities
    2. Response effectiveness and cleanup completion
    3. Long-term environmental impacts
    4. Patterns that might indicate disproportionate impacts
    5. Regulatory compliance and enforcement patterns
    
    Spill descriptions:
    {combined_ej_text}
    
    Provide an environmental justice assessment focusing on:
    - Whether vulnerable communities face more severe incidents
    - Quality of response and remediation
    - Policy recommendations for equitable environmental protection
    """
    
    return query_ollama(prompt)

def comprehensive_spill_analysis(csv_file):
    """Run complete analysis of spill data"""
    
    print("Loading spill data...")
    df = pd.read_csv(csv_file)
    
    print(f"Analyzing {len(df)} spill incidents...")
    
    # Basic demographic analysis
    demo_stats, ej_analysis = analyze_spill_demographics(df)
    
    # Root cause analysis (using existing categorizations)
    cause_counts, root_causes = analyze_root_causes(df)
    
    # Demographic patterns
    demo_patterns = demographic_spill_analysis(df)
    
    # LLM-based theme analysis
    print("Running LLM analysis on spill descriptions...")
    theme_analysis = analyze_spill_themes_llm(df, sample_size=100)
    
    # Environmental justice analysis
    print("Analyzing environmental justice implications...")
    ej_llm_analysis = analyze_environmental_justice(df, sample_descriptions=30)
    
    # Compile comprehensive results
    results = {
        'summary_statistics': {
            'total_incidents': len(df),
            'date_range': f"{df['Date of Discovery'].min()} to {df['Date of Discovery'].max()}",
            'counties_affected': df['county'].nunique(),
            'operators_involved': df['Operator'].nunique()
        },
        'demographic_statistics': demo_stats,
        'environmental_justice_analysis': ej_analysis,
        'root_cause_analysis': {
            'cause_counts': cause_counts,
            'top_root_causes': root_causes.to_dict()
        },
        'demographic_patterns': demo_patterns,
        'llm_theme_analysis': theme_analysis,
        'llm_environmental_justice': ej_llm_analysis
    }
    
    return results

def generate_policy_report(results):
    """Generate policy-focused summary using LLM"""
    
    # Create summary for LLM to process
    summary_text = f"""
    SPILL DATA ANALYSIS SUMMARY:
    
    Total Incidents: {results['summary_statistics']['total_incidents']}
    Date Range: {results['summary_statistics']['date_range']}
    
    DEMOGRAPHIC PATTERNS:
    - Average poverty rate in affected areas: {results['demographic_statistics']['avg_poverty_rate']:.1f}%
    - Average income: ${results['demographic_statistics']['avg_median_income']:,.0f}
    - Spills in high-poverty areas: {results['environmental_justice_analysis']['high_poverty_spills']}
    - Spills in minority communities: {results['environmental_justice_analysis']['minority_community_spills']}
    
    ROOT CAUSES:
    - Equipment failures: {results['root_cause_analysis']['cause_counts']['equipment_failure']}
    - Human error: {results['root_cause_analysis']['cause_counts']['human_error']}
    - Historical/unknown: {results['root_cause_analysis']['cause_counts']['historical_unknown']}
    
    THEME ANALYSIS:
    {results['llm_theme_analysis']}
    
    ENVIRONMENTAL JUSTICE ANALYSIS:
    {results['llm_environmental_justice']}
    """
    
    policy_prompt = f"""
    Based on this comprehensive spill data analysis, create a policy-focused executive summary.
    
    Data Summary:
    {summary_text}
    
    Provide:
    1. Key findings on environmental justice impacts
    2. Priority areas for regulatory attention
    3. Specific policy recommendations for prevention
    4. Recommendations for equitable enforcement
    5. Suggested regulatory changes based on patterns identified
    
    Format as an executive summary suitable for regulatory decision-makers and policy researchers.
    """
    
    return query_ollama(policy_prompt)

# Execute comprehensive analysis
if __name__ == "__main__":
    # Run the analysis
    results = comprehensive_spill_analysis('spills_with_demographics.csv')
    
    # Generate policy report
    print("\nGenerating policy-focused summary...")
    policy_report = generate_policy_report(results)
    
    # Save all results
    with open('comprehensive_spill_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('policy_executive_summary.txt', 'w') as f:
        f.write(policy_report)
    
    # Print key findings
    print("\n" + "="*60)
    print("COMPREHENSIVE SPILL ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"\nTotal incidents analyzed: {results['summary_statistics']['total_incidents']:,}")
    print(f"Counties affected: {results['summary_statistics']['counties_affected']}")
    print(f"Average poverty rate in affected areas: {results['demographic_statistics']['avg_poverty_rate']:.1f}%")
    print(f"Spills in high-poverty communities: {results['environmental_justice_analysis']['high_poverty_spills']:,}")
    print(f"Spills in minority communities: {results['environmental_justice_analysis']['minority_community_spills']:,}")
    
    print(f"\nRoot cause breakdown:")
    for cause, count in results['root_cause_analysis']['cause_counts'].items():
        print(f"  {cause.replace('_', ' ').title()}: {count:,}")
    
    print(f"\nResults saved to:")
    print(f"  - comprehensive_spill_analysis.json (detailed data)")
    print(f"  - policy_executive_summary.txt (executive summary)")
    
    print(f"\nPolicy Summary Preview:")
    print("="*40)
    print(policy_report[:500] + "...")
