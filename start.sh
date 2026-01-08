#!/bin/bash

# ✅ Set ngrok authtoken if provided (only needed first time or if using env variable)
if [ -n "$NGROK_AUTHTOKEN" ]; then
  echo "[+] Setting ngrok authtoken..."
  ngrok config add-authtoken "$NGROK_AUTHTOKEN"
fi

# ✅ Start ngrok in the background forwarding port 9050
echo "[+] Starting ngrok..."
ngrok http 9050 --log=stdout > /tmp/ngrok.log &

# ✅ Wait for ngrok to be ready (up to 10 seconds)
for i in {1..10}; do
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*' | grep 'ngrok' | head -n 1)
  if [ -n "$NGROK_URL" ]; then
    echo "------------------------------------------"
    echo "[+] Ngrok URL: $NGROK_URL"
    echo "$NGROK_URL" > current_ngrok_url.txt
    echo "[+] URL saved to current_ngrok_url.txt"
    echo "------------------------------------------"
    break
  fi
  sleep 1
done

# ✅ Start FastAPI server
echo "[+] Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 9050
