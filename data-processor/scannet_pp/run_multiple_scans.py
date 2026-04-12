import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import specific functions from the local pipeline script
from run_pipeline import run_pipeline, _resolve_data_dir

CONFIG = {
    "scan_ids": [
        # "09c1414f1b",
        # "0d2ee665be",
        "13c3e046d7",
        "1ada7a0617",
        "21d970d8de",
        "25f3b7a318",
        "27dd4da69e",
        "286b55a2bf",
        "31a2c91c43",
        "3864514494",
        "38d58a7a31",
        "3db0a1c8f3",
        "3e8bba0176",
        "3f15a9266d",
        "40aec5fffa",
        "45b0dac5e3",
        "5748ce6f01",
        "578511c8a9",
        "5942004064",
        "5eb31827b7",
        "5ee7c22ba0",
        "5f99900f09",
        "6115eddb86",
        "7831862f02",
        "7b6477cb95",
        "7bc286c1b6",
        "825d228aec",
        "9071e139d9",
        "99fa5c25e1",
        "a24f64f7fb",
        "a8bf42d646",
        "a980334473",
        "ac48a9b736",
        "acd95847c5",
        "b0a08200c9",
        "bcd2436daf",
        "bde1e479ad",
        "c49a8c6cff",
        "c4c04e6d6c",
        "c50d2d1d42",
        "c5439f4607",
        "cc5237fd77",
        "d755b3d9d8",
        "e398684d27",
        "e7af285f7d",
        "f2dc06b1d2",
        "f3685d06a9",
        "f3d64c30f8",
        "f9f95681fd",
        "fb5a96b1a2",
    ],
    "data_root": "/home/sagar/Repos/open-datasets/ScanNetPP/data/data",
    "output_root": "/home/sagar/Repos/openFLAME-repos/scan-to-map/outputs",
    "crop_source": "iphone"
}

def main():
    data_root = Path(CONFIG["data_root"]).resolve()
    output_root = Path(CONFIG["output_root"]).resolve()
    crop_source = CONFIG.get("crop_source", "iphone")

    for scan_id in CONFIG.get("scan_ids", []):
        print(f"\n{'='*60}")
        print(f"Processing scan: {scan_id}")
        print(f"{'='*60}\n")
        
        try:
            data_dir = _resolve_data_dir(scan_id, data_root=data_root)
            if not data_dir.is_dir():
                print(f"Skipping {scan_id}: Directory not found at {data_dir}")
                continue
                
            run_pipeline(
                data_dir=data_dir,
                output_root=output_root,
                crop_source=crop_source,
            )
            print(f"\nSuccessfully finished processing {scan_id}.")
        except Exception as e:
            print(f"\nError processing {scan_id}: {e}")

if __name__ == "__main__":
    main()
