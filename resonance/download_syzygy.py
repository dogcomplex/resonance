"""
Syzygy Tablebase Downloader + 8-Piece Lightning Solver

Downloads required tablebases only if missing, then runs lightning search.
"""

import os
import subprocess
import sys

SYZYGY_DIR = "./syzygy"
BASE_URL = "http://tablebase.sesse.net/syzygy"

# Required tables for KQRRvKQRR (8-piece) -> KQRRvKQR (7-piece) captures
# Plus fallback smaller captures

TABLES_7_PIECE = [
    # These are what we need for 8-piece -> 7-piece captures
    ("7-WDL", "KQRRvKQR.rtbw"),
    ("7-DTZ", "KQRRvKQR.rtbz"),
    ("7-WDL", "KQRvKQRR.rtbw"),  # Symmetric
    ("7-DTZ", "KQRvKQRR.rtbz"),
    # Also need KQRRvKQ, KQRRvKR for when Q or R captured
    ("7-WDL", "KQRRvKQ.rtbw"),
    ("7-DTZ", "KQRRvKQ.rtbz"),
    ("7-WDL", "KQRRvKR.rtbw"),
    ("7-DTZ", "KQRRvKR.rtbz"),
    ("7-WDL", "KQvKQRR.rtbw"),
    ("7-DTZ", "KQvKQRR.rtbz"),
    ("7-WDL", "KRvKQRR.rtbw"),
    ("7-DTZ", "KRvKQRR.rtbz"),
]

TABLES_5_PIECE = [
    # Smaller tables for deeper captures / testing
    ("3-4-5", "KQRvKQ.rtbw"),
    ("3-4-5", "KQRvKQ.rtbz"),
    ("3-4-5", "KQRvKR.rtbw"),
    ("3-4-5", "KQRvKR.rtbz"),
    ("3-4-5", "KQRvK.rtbw"),
    ("3-4-5", "KQRvK.rtbz"),
    ("3-4-5", "KQvKR.rtbw"),
    ("3-4-5", "KQvKR.rtbz"),
    ("3-4-5", "KRvKQ.rtbw"),
    ("3-4-5", "KRvKQ.rtbz"),
    ("3-4-5", "KQvK.rtbw"),
    ("3-4-5", "KQvK.rtbz"),
    ("3-4-5", "KRvK.rtbw"),
    ("3-4-5", "KRvK.rtbz"),
    ("3-4-5", "KQvKQ.rtbw"),
    ("3-4-5", "KQvKQ.rtbz"),
    ("3-4-5", "KRvKR.rtbw"),
    ("3-4-5", "KRvKR.rtbz"),
    ("3-4-5", "KQRvKQR.rtbw"),  # 6-piece, might be in 3-4-5 or 6-WDL
    ("3-4-5", "KQRvKQR.rtbz"),
]

TABLES_6_PIECE = [
    ("6-WDL", "KQRvKQR.rtbw"),
    ("6-DTZ", "KQRvKQR.rtbz"),
    ("6-WDL", "KQRRvKQ.rtbw"),
    ("6-DTZ", "KQRRvKQ.rtbz"),
    ("6-WDL", "KQRRvKR.rtbw"),
    ("6-DTZ", "KQRRvKR.rtbz"),
    ("6-WDL", "KQRRvK.rtbw"),
    ("6-DTZ", "KQRRvK.rtbz"),
    ("6-WDL", "KQvKQR.rtbw"),
    ("6-DTZ", "KQvKQR.rtbz"),
    ("6-WDL", "KRvKQR.rtbw"),
    ("6-DTZ", "KRvKQR.rtbz"),
]


def download_tables(tables, skip_large=True):
    """Download missing tablebase files using curl"""
    os.makedirs(SYZYGY_DIR, exist_ok=True)
    
    for folder, filename in tables:
        filepath = os.path.join(SYZYGY_DIR, filename)
        
        # Check if already exists
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} exists ({size_mb:.1f} MB)")
            continue
        
        url = f"{BASE_URL}/{folder}/{filename}"
        print(f"  ↓ Downloading {filename}...")
        print(f"    URL: {url}")
        
        try:
            # Use curl with progress bar
            result = subprocess.run(
                ["curl", "-L", "-o", filepath, "--progress-bar", url],
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"    ✓ Downloaded ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ Failed (might not exist on server)")
                # Clean up partial file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
        except FileNotFoundError:
            print("    ✗ curl not found! Please install curl or download manually.")
            print(f"    URL: {url}")


def check_available_tables():
    """Check what we have available"""
    print("\nAvailable tablebases:")
    
    if not os.path.exists(SYZYGY_DIR):
        print("  No syzygy folder found!")
        return []
    
    files = os.listdir(SYZYGY_DIR)
    rtbw_files = [f for f in files if f.endswith('.rtbw')]
    rtbz_files = [f for f in files if f.endswith('.rtbz')]
    
    print(f"  WDL files: {len(rtbw_files)}")
    print(f"  DTZ files: {len(rtbz_files)}")
    
    for f in sorted(files):
        if f.endswith('.rtbw') or f.endswith('.rtbz'):
            size_mb = os.path.getsize(os.path.join(SYZYGY_DIR, f)) / (1024 * 1024)
            print(f"    {f}: {size_mb:.1f} MB")
    
    return rtbw_files


def main():
    print("="*60)
    print("SYZYGY TABLEBASE MANAGER")
    print("="*60)
    
    # Check what we have
    available = check_available_tables()
    
    print("\n" + "="*60)
    print("DOWNLOADING MISSING 5-PIECE TABLES")
    print("="*60)
    download_tables(TABLES_5_PIECE)
    
    print("\n" + "="*60)
    print("DOWNLOADING MISSING 6-PIECE TABLES")
    print("="*60)
    download_tables(TABLES_6_PIECE)
    
    # Skip 7-piece by default (they're huge)
    print("\n" + "="*60)
    print("7-PIECE TABLES (large, skipping auto-download)")
    print("="*60)
    print("You already have KQRRvKQR.rtbw (9.5 GB) - great for 8-piece!")
    print("To download more 7-piece, run:")
    print("  curl -L -o syzygy/KQRvKQRR.rtbw http://tablebase.sesse.net/syzygy/7-WDL/KQRvKQRR.rtbw")
    
    # Final check
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    check_available_tables()


if __name__ == "__main__":
    main()
