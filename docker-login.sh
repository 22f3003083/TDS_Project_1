#!/bin/bash
# Simple script to login to Docker

# Prompt for username
read -p "Enter your Docker username: " username

# Prompt for password (input hidden)
read -sp "Enter your Docker password: " password
echo  # Newline after password input

# Log in to Docker Hub (or replace with your registry URL)
docker login --username "$username" --password "$password"
