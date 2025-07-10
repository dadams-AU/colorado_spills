import requests
import time
import json

def test_ollama_step_by_step():
    """Test each step of Ollama communication"""
    
    print("=== OLLAMA DEBUG TEST ===")
    
    # Step 1: Test basic connection
    print("1. Testing basic connection...")
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        print(f"   ✓ Connection successful: {response.status_code}")
        models = response.json().get('models', [])
        print(f"   ✓ Found {len(models)} models")
        for model in models:
            print(f"     - {model['name']}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False
    
    # Step 2: Test simple generation
    print("\n2. Testing simple generation...")
    simple_prompt = "Hello"
    
    try:
        print("   Sending simple request...")
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral:7b',
                'prompt': simple_prompt,
                'stream': False,
                'options': {'num_predict': 50}  # Limit to 50 tokens
            },
            timeout=30  # 30 second timeout
        )
        
        elapsed = time.time() - start_time
        print(f"   ✓ Request completed in {elapsed:.1f} seconds")
        print(f"   ✓ Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print(f"   ✓ Response received: {result['response'][:100]}...")
                return True
            else:
                print(f"   ✗ Unexpected response format: {result}")
                return False
        else:
            print(f"   ✗ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"   ✗ Request failed: {e}")
        return False
    
    print("\n3. Testing with longer prompt...")
    longer_prompt = "Analyze this data and provide insights: " + "data point " * 50
    
    try:
        print("   Sending longer request...")
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral:7b',
                'prompt': longer_prompt,
                'stream': False,
                'options': {'num_predict': 100}
            },
            timeout=60
        )
        
        elapsed = time.time() - start_time
        print(f"   ✓ Longer request completed in {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Response length: {len(result.get('response', ''))}")
            return True
        else:
            print(f"   ✗ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Longer request failed: {e}")
        return False

def test_environmental_justice_prompt():
    """Test with a prompt similar to your actual use case"""
    
    print("\n=== TESTING EJ ANALYSIS PROMPT ===")
    
    # Simplified version of your actual prompt
    test_prompt = """Analyze these environmental justice findings:

STATISTICAL TESTS:
- Poverty over-representation: 2.3x expected (p=0.001)
- Major spills z-test p-value: 0.023
- Minority communities ratio: 1.8x

SPATIAL PATTERNS:
- 5 spatial clusters identified
- Max density: 12 spills per grid cell

Provide a 200-word academic interpretation focusing on environmental justice implications."""
    
    try:
        print("Sending EJ analysis prompt...")
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral:7b',
                'prompt': test_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 300
                }
            },
            timeout=120  # 2 minute timeout
        )
        
        elapsed = time.time() - start_time
        print(f"✓ EJ prompt completed in {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            analysis = result.get('response', '')
            print(f"✓ Analysis generated ({len(analysis)} characters)")
            print(f"Preview: {analysis[:200]}...")
            return True
        else:
            print(f"✗ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ EJ prompt failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama integration...")
    
    if test_ollama_step_by_step():
        print("\n✓ Basic Ollama tests passed!")
        test_environmental_justice_prompt()
    else:
        print("\n✗ Basic Ollama tests failed!")
    
    print("\n=== DEBUG COMPLETE ===")