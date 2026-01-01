#!/usr/bin/env python3
"""
StudyMate Setup Validator
Tests the basic functionality and dependencies.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'langchain',
        'langchain_openai',
        'langchain_community',
        'chromadb',
        'pypdf',
        'openai'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    return missing_packages

def check_environment():
    """Check environment setup."""
    print("\n🔍 Checking environment setup...")

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            print("✅ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not set or using placeholder")
    else:
        print("❌ .env file not found")

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} (supported)")
    else:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} (minimum 3.8 recommended)")

def check_files():
    """Check if all necessary files exist."""
    print("\n📁 Checking project files...")

    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]

    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")

def main():
    print("🧪 StudyMate Setup Validator")
    print("=" * 30)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing = check_dependencies()

    # Check environment
    check_environment()

    # Check files
    check_files()

    # Summary
    print("\n" + "=" * 30)
    if not missing:
        print("🎉 All dependencies installed!")
        print("🚀 You can run: streamlit run app.py")
    else:
        print("⚠️  Missing dependencies. Run: pip install -r requirements.txt")
        print(f"Missing: {', '.join(missing)}")

    print("\n📖 For detailed setup instructions, see README.md")

if __name__ == "__main__":
    main()