# scripts/schema_utils.py

def validate_input_schema(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    print("✅ Input schema validated.")
