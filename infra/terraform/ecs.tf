resource "aws_ecs_cluster" "this" {
  name = "${var.project}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_cloudwatch_log_group" "service" {
  for_each          = local.services
  name              = "/ecs/${var.project}-${each.key}"
  retention_in_days = 14
}

# The retrieval service has no public entrypoint (see networking.tf), so the
# agent task discovers it by internal DNS via AWS Cloud Map rather than a
# second internal load balancer -- ECS Fargate tasks get a fresh private IP
# on every deployment, so something has to keep "retrieval.ragent.local"
# pointed at whichever IP is currently healthy.
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = "${var.project}.local"
  vpc  = local.vpc_id
}

resource "aws_service_discovery_service" "retrieval" {
  name = "retrieval"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.internal.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# --- IAM ---------------------------------------------------------------- #

data "aws_iam_policy_document" "ecs_task_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${var.project}-ecs-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- Retrieval service: task definition + ECS service ------------------- #

resource "aws_ecs_task_definition" "retrieval" {
  family                   = "${var.project}-retrieval"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = local.services.retrieval.cpu
  memory                   = local.services.retrieval.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "retrieval"
      image     = "${aws_ecr_repository.service["retrieval"].repository_url}:${local.services.retrieval.tag}"
      essential = true
      portMappings = [
        { containerPort = local.services.retrieval.port, protocol = "tcp" }
      ]
      environment = [
        { name = "EMBEDDER_BACKEND", value = "auto" },
        { name = "VECTORSTORE_BACKEND", value = "auto" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.service["retrieval"].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "retrieval"
        }
      }
    }
  ])

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_ecs_service" "retrieval" {
  name            = "${var.project}-retrieval"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.retrieval.arn
  desired_count   = local.services.retrieval.count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.subnet_ids
    security_groups  = [aws_security_group.retrieval_service.id]
    assign_public_ip = true # default VPC has no NAT gateway; see networking.tf note
  }

  service_registries {
    registry_arn = aws_service_discovery_service.retrieval.arn
  }
}

# --- Agent service: task definition + ECS service ------------------------ #

resource "aws_ecs_task_definition" "agent" {
  family                   = "${var.project}-agent"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = local.services.agent.cpu
  memory                   = local.services.agent.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "agent"
      image     = "${aws_ecr_repository.service["agent"].repository_url}:${local.services.agent.tag}"
      essential = true
      portMappings = [
        { containerPort = local.services.agent.port, protocol = "tcp" }
      ]
      environment = [
        {
          name  = "RETRIEVAL_SERVICE_URL"
          value = "http://retrieval.${aws_service_discovery_private_dns_namespace.internal.name}:${local.services.retrieval.port}"
        },
        { name = "LLM_BACKEND", value = "auto" },
        { name = "AGENT_MAX_STEPS", value = "6" },
        # NOTE: plain `environment` entries land in plaintext in the task
        # definition (visible via DescribeTaskDefinition to anyone with ECS
        # read access). A real production deployment should use the
        # `secrets` block instead, pointing at an AWS Secrets Manager ARN or
        # SSM Parameter Store path, so the key is only resolved into the
        # container's environment at task launch and never appears in the
        # task definition itself. Left as a plain variable here to keep this
        # stack `terraform apply`-able with zero pre-existing AWS resources;
        # see the README's infrastructure section for the secrets-manager
        # version of this block.
        { name = "OPENAI_API_KEY", value = var.openai_api_key },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.service["agent"].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "agent"
        }
      }
    }
  ])

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_ecs_service" "agent" {
  name            = "${var.project}-agent"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = local.services.agent.count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.subnet_ids
    security_groups  = [aws_security_group.agent_service.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent.arn
    container_name   = "agent"
    container_port   = local.services.agent.port
  }

  depends_on = [aws_lb_listener.agent_http]
}
