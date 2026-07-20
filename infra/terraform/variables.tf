variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Short name used to prefix/tag every resource this stack creates."
  type        = string
  default     = "ragent"
}

variable "environment" {
  description = "Deployment environment name (e.g. dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "retrieval_image_tag" {
  description = "Image tag to deploy for the retrieval service (pushed to the ECR repo this stack creates)."
  type        = string
  default     = "latest"
}

variable "agent_image_tag" {
  description = "Image tag to deploy for the agent orchestrator service."
  type        = string
  default     = "latest"
}

variable "retrieval_desired_count" {
  description = "Number of retrieval-service tasks. See infra/k8s/retrieval.yaml for why this stays at 1 by default -- the service holds its vector index in-process."
  type        = number
  default     = 1
}

variable "agent_desired_count" {
  description = "Number of agent-service tasks. Stateless, safe to scale horizontally."
  type        = number
  default     = 2
}

variable "agent_cpu" {
  description = "Fargate task CPU units for the agent service (256 = .25 vCPU)."
  type        = number
  default     = 256
}

variable "agent_memory" {
  description = "Fargate task memory (MiB) for the agent service."
  type        = number
  default     = 512
}

variable "retrieval_cpu" {
  description = "Fargate task CPU units for the retrieval service."
  type        = number
  default     = 512
}

variable "retrieval_memory" {
  description = "Fargate task memory (MiB) for the retrieval service. Higher than the agent service since it holds the embedding model and vector index in memory."
  type        = number
  default     = 1024
}

variable "openai_api_key" {
  description = "Optional OpenAI-compatible API key for the agent service's real LLM backend. Leave empty to run against StubLLM. Never commit a real value -- pass via TF_VAR_openai_api_key or a secrets manager in CI."
  type        = string
  default     = ""
  sensitive   = true
}
