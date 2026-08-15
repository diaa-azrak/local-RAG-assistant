from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    print("=== Week 1: Foundry Local Verification ===")
    
    # 1. Initialize Foundry Local Manager
    config = Configuration(app_name="LocalRAGAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2. Query available catalog models
    print("\n[1/3] Querying local catalog models...")
    models = manager.catalog.list_models()
    print(f"Catalog loaded successfully. Total models available: {len(models)}")

    # 3. Download & load a lightweight model
    # Aliases auto-select optimal execution providers (CPU, GPU, or NPU)
    model_alias = "qwen2.5-0.5b"
    print(f"\n[2/3] Downloading/Loading model alias: '{model_alias}'...")
    
    selected_model = manager.catalog.get_model(model_alias)
    selected_model.download()
    selected_model.load()
    print("Model loaded into device memory successfully!")

    # 4. Perform local text generation
    print("\n[3/3] Generating local inference response...")
    chat_client = selected_model.create_chat_client()
    
    response = chat_client.complete_chat([
        {"role": "user", "content": "Hello! Confirm you are running on-device with zero internet connection."}
    ])

    print("\n" + "=" * 50)
    print("LOCAL MODEL RESPONSE:")
    print(response.choices[0].message.content)
    print("=" * 50)
    print("\nWeek 1 Milestone Achieved: Local execution verified!")

if __name__ == "__main__":
    main()