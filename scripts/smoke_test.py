from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the Eagle AI tagger service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service base URL")
    parser.add_argument("--image-path", help="Optional image path for /tag")
    args = parser.parse_args()

    health = requests.get(f"{args.base_url}/healthz", timeout=10)
    health.raise_for_status()
    print("healthz")
    print(json.dumps(health.json(), ensure_ascii=False, indent=2))

    if args.image_path:
        response = requests.post(
            f"{args.base_url}/tag",
            json={"image_path": args.image_path},
            timeout=60,
        )
        response.raise_for_status()
        print("tag")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
