"""
local_test.py
=============
Run this BEFORE deploying to make sure your stats look correct.

Usage:
    python local_test.py
"""

import json
import base64
import requests

# ── 1. Generate a tiny fake WAV in memory (just for format testing) ──
import io, wave, struct, math

def make_sine_wav(freq=440, duration=0.5, sample_rate=16000) -> bytes:
    """Create a minimal valid WAV file as bytes."""
    n_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack('<h', val))
    return buf.getvalue()


def test_api(base_url: str = "http://localhost:5000"):
    print(f"Testing API at {base_url} ...\n")

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print(f"[Health] {r.status_code}: {r.json()}")
    except Exception as e:
        print(f"[Health] FAILED: {e}")
        return

    # POST with fake audio
    fake_wav   = make_sine_wav()
    fake_b64   = base64.b64encode(fake_wav).decode()
    payload    = {"audio_id": "q0", "audio_base64": fake_b64}

    try:
        r = requests.post(base_url + "/", json=payload, timeout=30)
        print(f"\n[POST /] Status: {r.status_code}")
        result = r.json()
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

        # Check required keys
        required = ["rows","columns","mean","std","variance","min","max",
                    "median","mode","range","allowed_values","value_range","correlation"]
        missing = [k for k in required if k not in result]
        if missing:
            print(f"\n⚠️  Missing keys: {missing}")
        else:
            print("\n✅ All required keys present!")

    except Exception as e:
        print(f"[POST /] FAILED: {e}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_api(url)
