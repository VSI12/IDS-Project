# Deploying an Intrusion Detection System to AWS 

## Project Overview
This Project demonstrates how to build and containerize a flask web-application with docker and deploy it to AWS. This architecture ensures a secure, highly available, fault tolerant and scalable build by leveraging various AWS architectures.

## Architecture
- **Virtual Private Cloud (VPC)**: Configured with public and private subnets across two availability zones for high availability and security.
- **Interget Gateway**: Enables communication between the VPC and the internet
- **VPC Endpoints**: The VPC endpoints enable the ECS tasks in the private subnet to access certain resources.
- 
  ![IDS-Architecture (2)](https://github.com/user-attachments/assets/fda8ad0d-d0b9-448c-92d7-ff1a46760c3a)
