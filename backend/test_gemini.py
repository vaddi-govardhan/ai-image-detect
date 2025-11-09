#!/usr/bin/env python3
"""
Test script to verify Gemini API connection
Run this in your backend directory
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_connection():
    """Test if Gemini API is properly configured"""
    
    print("=" * 50)
    print("GEMINI API CONNECTION TEST")
    print("=" * 50)
    
    # Check if API key exists
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("📝 Please add it to your .env file:")
        print("   GOOGLE_API_KEY=your_actual_key_here")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Try to configure and use Gemini
    try:
        print("\n🔗 Configuring Gemini API...")
        genai.configure(api_key=api_key)
        
        # List available models first
        print("📋 Listing available models...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"   ✓ {m.name}")
        
        if not available_models:
            raise Exception("No models available for content generation")
        
        # Try models in order of preference
        model_priority = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro',
            'models/gemini-pro-vision'
        ]
        
        model_to_use = None
        for preferred_model in model_priority:
            if preferred_model in available_models:
                model_to_use = preferred_model
                break
        
        if not model_to_use:
            model_to_use = available_models[0]
        
        print(f"\n🤖 Testing model: {model_to_use}")
        model = genai.GenerativeModel(model_to_use)
        
        # Test simple text generation
        print("💬 Sending test prompt...")
        response = model.generate_content("Say 'Hello! Gemini is working!' in a friendly way.")
        
        print("\n✅ SUCCESS! Gemini Response:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔍 Possible issues:")
        print("   1. Invalid API key")
        print("   2. API key doesn't have Gemini API enabled")
        print("   3. Network connection issues")
        print("   4. google-generativeai package not installed")
        print("\n💡 Solutions:")
        print("   - Verify your API key at: https://makersuite.google.com/app/apikey")
        print("   - Install package: pip install google-generativeai")
        return False

if __name__ == "__main__":
    test_gemini_connection()