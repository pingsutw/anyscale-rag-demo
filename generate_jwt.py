#!/usr/bin/env python3
import sys
import time

import jwt

# Get PEM file path
if len(sys.argv) > 1:
    pem = sys.argv[1]
else:
    pem = input("Enter path of private PEM file: ")

# Get the Client ID
if len(sys.argv) > 2:
    client_id = sys.argv[2]
else:
    client_id = input("Enter your Client ID: ")

# Open PEM
with open(pem, 'rb') as pem_file:
    signing_key = pem_file.read()

payload = {
    # Issued at time
    'iat': int(time.time()),
    # JWT expiration time (10 minutes maximum)
    'exp': int(time.time()) + 600,

    # GitHub App's client ID
    'iss': client_id
}

# Create JWT
encoded_jwt = jwt.encode(payload, signing_key, algorithm='RS256')

print(f"JWT:  {encoded_jwt}")

"""
To Generate GITHUB_BOT_ACCESS_TOKEN, run the following command:

$ python generate_jwt.py
$ curl --request POST \
    --url "https://api.github.com/app/installations/54078058/access_tokens" \
    --header "Accept: application/vnd.github+json" \
    --header "Authorization: Bearer $JWT" \
    --header "X-GitHub-Api-Version: 2022-11-28"

To leave comments on GitHub issues, run the following command:
$ curl -H "Authorization: Bearer $ACCESS_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     -d '{"body":"test"}' \
     https://api.github.com/repos/pingsutw/flyte-action/issues/1/comments
"""
