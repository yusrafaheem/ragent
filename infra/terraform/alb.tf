# Public entrypoint: only the agent service sits behind this ALB. The
# retrieval service is intentionally not exposed here -- see
# aws_security_group.retrieval_service in networking.tf.

resource "aws_lb" "agent" {
  name               = "${var.project}-agent-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = local.subnet_ids

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_lb_target_group" "agent" {
  name        = "${var.project}-agent-tg"
  port        = local.services.agent.port
  protocol    = "HTTP"
  vpc_id      = local.vpc_id
  target_type = "ip" # required for awsvpc-networked Fargate tasks

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 15
    timeout             = 5
    matcher             = "200"
  }

  tags = { Project = var.project, Environment = var.environment }
}

resource "aws_lb_listener" "agent_http" {
  load_balancer_arn = aws_lb.agent.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agent.arn
  }
}
