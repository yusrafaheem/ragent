locals {
  services = {
    retrieval = {
      port   = 8001
      cpu    = var.retrieval_cpu
      memory = var.retrieval_memory
      count  = var.retrieval_desired_count
      tag    = var.retrieval_image_tag
    }
    agent = {
      port   = 8000
      cpu    = var.agent_cpu
      memory = var.agent_memory
      count  = var.agent_desired_count
      tag    = var.agent_image_tag
    }
  }
}

resource "aws_ecr_repository" "service" {
  for_each = local.services

  name                 = "${var.project}-${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project
    Environment = var.environment
    Service     = each.key
  }
}

resource "aws_ecr_lifecycle_policy" "service" {
  for_each   = local.services
  repository = aws_ecr_repository.service[each.key].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 14 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 14
        }
        action = { type = "expire" }
      }
    ]
  })
}
