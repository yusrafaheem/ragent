# Uses the account's default VPC/subnets rather than provisioning a custom
# one, to keep this stack deployable in a fresh AWS account with a single
# `terraform apply` and no networking prerequisites. A real production
# rollout would swap this data source for a purpose-built VPC module with
# private subnets for the ECS tasks and NAT egress -- everything downstream
# (security groups, ECS service network_configuration, ALB subnets) only
# depends on `local.vpc_id` and `local.subnet_ids`, so that swap doesn't
# touch any other file.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  vpc_id     = data.aws_vpc.default.id
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_security_group" "alb" {
  name        = "${var.project}-alb"
  description = "Allow inbound HTTP from the internet to the agent service ALB"
  vpc_id      = local.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_security_group" "agent_service" {
  name        = "${var.project}-agent-service"
  description = "Agent orchestrator tasks: accept traffic from the ALB only"
  vpc_id      = local.vpc_id

  ingress {
    description     = "From ALB"
    from_port       = local.services.agent.port
    to_port         = local.services.agent.port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_security_group" "retrieval_service" {
  name        = "${var.project}-retrieval-service"
  description = "Retrieval tasks: internal only, reachable from the agent service exclusively -- never from the public ALB"
  vpc_id      = local.vpc_id

  ingress {
    description     = "From agent service"
    from_port       = local.services.retrieval.port
    to_port         = local.services.retrieval.port
    protocol        = "tcp"
    security_groups = [aws_security_group.agent_service.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = var.project, Environment = var.environment }
}
