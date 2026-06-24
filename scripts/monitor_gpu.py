#!/usr/bin/env python3
import argparse
import subprocess
import time
from datetime import datetime
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Log GPU metrics (temperature, power, utilization) to a CSV file.")
    parser.add_argument("--output", "-o", type=str, default="gpu_monitor.csv",
                        help="Path to the output CSV file (default: gpu_monitor.csv)")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                        help="Logging interval in seconds (default: 1.0)")
    args = parser.parse_args()

    # Expand user/relative paths
    output_path = os.path.abspath(args.output)
    print(f"Logging GPU metrics to: {output_path}")
    print(f"Polling interval: {args.interval}s")
    print("Press Ctrl+C to stop.")

    # Write header if file does not exist
    if not os.path.exists(output_path):
        try:
            with open(output_path, "w") as f:
                f.write("timestamp,gpu_temp_c,power_draw_w,utilization_gpu_pct,utilization_mem_pct\n")
        except Exception as e:
            print(f"Error creating output file: {e}", file=sys.stderr)
            sys.exit(1)

    # Command to run
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits"
    ]

    try:
        while True:
            t_start = time.time()
            
            # Query nvidia-smi
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"Error running nvidia-smi: {res.stderr.strip()}", file=sys.stderr)
                time.sleep(max(0.1, args.interval))
                continue
                
            # Process output
            # Format is: temp, power, util_gpu, util_mem
            # e.g.: "81, 270.47, 99, 33"
            data = res.stdout.strip()
            if not data:
                continue
                
            timestamp = datetime.now().isoformat()
            log_line = f"{timestamp},{data}\n"
            
            # Append to file
            with open(output_path, "a") as f:
                f.write(log_line)
                
            # Print to stdout for visibility
            print(f"[{timestamp}] Temp: {data.split(',')[0]}°C | Power: {data.split(',')[1]}W | GPU Util: {data.split(',')[2]}% | Mem Util: {data.split(',')[3]}%")
            
            # Adjust sleep time to match interval
            elapsed = time.time() - t_start
            sleep_time = max(0.01, args.interval - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
