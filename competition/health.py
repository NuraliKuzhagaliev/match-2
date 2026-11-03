import json
import sys
import uuid
from competition.contract import rpc


def main():
    try:
        response = rpc({"operation": "health", "request_id": uuid.uuid4().hex}, timeout=3.0)
        print(json.dumps(response, ensure_ascii=False))
        return 0 if response.get("ready") else 1
    except (OSError, ValueError, RuntimeError, ConnectionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
