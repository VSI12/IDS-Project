# Deploying an Intrusion Detection System to AWS 

## Project Overview
This Project demonstrates how to build and containerize a flask web-application with docker and deploy it to AWS. This architecture ensures a secure, highly available, fault tolerant and scalable build by leveraging various AWS architectures.

## Architecture
- **Virtual Private Cloud (VPC)**: Configured with public and private subnets across two availability zones for high availability and security.
- **Interget Gateway**: Enables communication between the VPC and the internet
- **VPC Endpoints**: The VPC endpoints enable the ECS tasks in the private subnet to access certain resources.
- **Application Load Balancer(ALB)**: Configured to forward traffic to the ECS tasks, through listeners and target groups and distributes the incoming traffic across the ECS tasks.
- **Elastic Container Registry**: Host the docker image
- **Elastic Container Service(ECS)**: creates the ECS cluster that hosts the Fargate service tasks
- **Code Pipeline**: Creates a CI/CD pipeline using codebuild for seamless integration and deployments.

## AWS Architecture Diagram
  ![IDS-Architecture (2)](https://github.com/user-attachments/assets/fda8ad0d-d0b9-448c-92d7-ff1a46760c3a)


## Deployment Steps
I have layed out the deployment steps in very high detail in my dev.to blog post. You're welcome to check it out.
- **DEV.TO blog post**: [BLOG POST](https://dev.to/non-existent/deploying-a-flask-based-intrusion-detection-system-to-aws-ecs-with-cicd-4pgm)
