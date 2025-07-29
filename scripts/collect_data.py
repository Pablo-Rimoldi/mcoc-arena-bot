# Data collection script
from src.data_collection.data_collector import MCOCDataCollector
import time


print("Starting MCOC Data Collector (10 second test)...")
try:
    time.sleep(4)
    collector = MCOCDataCollector()
    print(f"Saving to: {collector.get_output_dir()}")
    print("Press W, A, S, D, SPACE while testing...")
    collector.start()
    time.sleep(30000)  # Run for 10 seconds
    collector.stop()
    print("Test completed!")

except KeyboardInterrupt:
    print("\nStopping...")
    collector.stop()
except Exception as e:
    print(f"Error: {e}")
