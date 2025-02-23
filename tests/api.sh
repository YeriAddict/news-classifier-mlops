#!/bin/bash
set -e

echo "Waiting for API to start..."
for i in {1..30}; do
  if curl -s http://localhost:8000/docs >/dev/null; then
    echo "API is up!"
    break
  fi
  echo "Waiting for API..."
  sleep 2
done

status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs)
if [ "$status" -ne 200 ]; then
  echo "Expected 200 OK from /docs but got $status"
  exit 1
fi
echo "/docs endpoint is returning 200 OK."

response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"text": "Napoleon is British"}' \
    http://localhost:8000/api/predict)
echo "Prediction response: $response"
