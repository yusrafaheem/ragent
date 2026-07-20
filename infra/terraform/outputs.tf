output "agent_service_url" {
  description = "Public HTTP endpoint for the agent service (POST /ask, GET /health)."
  value       = "http://${aws_lb.agent.dns_name}"
}

output "ecr_repository_urls" {
  description = "Push images here before the ECS services will start successfully."
  value       = { for k, repo in aws_ecr_repository.service : k => repo.repository_url }
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "cloudwatch_log_groups" {
  value = { for k, lg in aws_cloudwatch_log_group.service : k => lg.name }
}
