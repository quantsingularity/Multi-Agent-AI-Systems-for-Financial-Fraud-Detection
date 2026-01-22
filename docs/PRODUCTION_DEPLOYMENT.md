# Production Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Steps](#deployment-steps)
4. [Scaling Guidelines](#scaling-guidelines)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Disaster Recovery](#disaster-recovery)
7. [Security](#security)
8. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### Required Infrastructure

- **Kubernetes Cluster**: v1.24+ (GKE, EKS, or AKS recommended)
- **CPU**: Minimum 16 vCPUs for production
- **Memory**: Minimum 32 GB RAM
- **Storage**: 100 GB+ SSD for databases and models
- **Network**: 10 Gbps+ recommended for high throughput

### Required Tools

```bash
# Install required CLI tools
kubectl >= 1.24
helm >= 3.10
docker >= 20.10
```

### Cloud Provider Setup

#### AWS (EKS)

```bash
# Create EKS cluster
eksctl create cluster \
  --name fraud-detection-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 20 \
  --managed
```

#### GCP (GKE)

```bash
# Create GKE cluster
gcloud container clusters create fraud-detection-prod \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 20 \
  --enable-autorepair \
  --enable-autoupgrade
```

#### Azure (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group fraud-detection-rg \
  --name fraud-detection-prod \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20
```

---

## Infrastructure Setup

### 1. Create Namespace

```bash
kubectl create namespace fraud-detection
kubectl config set-context --current --namespace=fraud-detection
```

### 2. Install Dependencies

#### Install Cert-Manager (for TLS)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

#### Install NGINX Ingress Controller

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.metrics.enabled=true
```

#### Install Prometheus Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring-values.yaml
```

### 3. Configure Secrets

Create secrets file (`secrets.yaml`):

```bash
# Generate secure passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)

# Create secrets
kubectl create secret generic fraud-detection-secrets \
  --from-literal=postgres-password=$POSTGRES_PASSWORD \
  --from-literal=redis-password=$REDIS_PASSWORD \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --namespace fraud-detection
```

---

## Deployment Steps

### Method 1: Helm Chart (Recommended)

```bash
# Add Helm repository
helm repo add fraud-detection https://your-helm-repo.com

# Install with production values
helm install fraud-detection fraud-detection/fraud-detection \
  --namespace fraud-detection \
  --values values-production.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n fraud-detection
kubectl get svc -n fraud-detection
```

### Method 2: Direct Kubernetes Manifests

```bash
# Apply in order
kubectl apply -f k8s/configmap-secrets.yaml
kubectl apply -f k8s/redis-postgres.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/ingress-policies.yaml

# Wait for rollout
kubectl rollout status deployment/fraud-detection-api -n fraud-detection
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n fraud-detection -w

# Check logs
kubectl logs -f deployment/fraud-detection-api -n fraud-detection

# Test health endpoint
kubectl port-forward svc/fraud-detection-api 8000:80 -n fraud-detection
curl http://localhost:8000/health
```

### 4. Load Initial Model

```bash
# Copy model to persistent volume
kubectl cp ./models/production_model.pkl \
  fraud-detection-api-xxxxx:/app/models/production_model.pkl

# Verify model loaded
kubectl exec -it deployment/fraud-detection-api -n fraud-detection -- \
  ls -lh /app/models/
```

---

## Scaling Guidelines

### Capacity Planning

| Transaction Volume   | Replicas | CPU/Pod | Memory/Pod | Estimated TPS |
| -------------------- | -------- | ------- | ---------- | ------------- |
| 1M/day (~12 TPS)     | 3        | 500m    | 1Gi        | 100 TPS       |
| 10M/day (~120 TPS)   | 10       | 1000m   | 2Gi        | 1K TPS        |
| 100M/day (~1.2K TPS) | 30       | 2000m   | 4Gi        | 10K TPS       |
| 1B/day (~12K TPS)    | 100      | 2000m   | 4Gi        | 100K TPS      |

### Auto-scaling Configuration

```yaml
# Horizontal Pod Autoscaler (HPA)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-api-hpa
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Manual Scaling

```bash
# Scale up
kubectl scale deployment fraud-detection-api --replicas=10 -n fraud-detection

# Scale down (carefully!)
kubectl scale deployment fraud-detection-api --replicas=5 -n fraud-detection
```

### Cluster Auto-scaling

Enable node auto-scaling to handle pod scaling:

```bash
# AWS EKS
eksctl scale nodegroup --cluster fraud-detection-prod \
  --nodes 10 --nodes-min 3 --nodes-max 20

# GKE
gcloud container clusters update fraud-detection-prod \
  --enable-autoscaling \
  --min-nodes 3 --max-nodes 20

# AKS
az aks update \
  --resource-group fraud-detection-rg \
  --name fraud-detection-prod \
  --update-cluster-autoscaler \
  --min-count 3 --max-count 20
```

---

## Monitoring & Alerting

### Access Monitoring Dashboards

```bash
# Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default credentials: admin / prom-operator
```

### Key Metrics to Monitor

1. **Performance Metrics**
   - Prediction latency (P50, P95, P99)
   - Throughput (requests/second)
   - Error rate

2. **Model Metrics**
   - Precision, Recall, F1 Score
   - False positive/negative rates
   - Concept drift indicators

3. **Infrastructure Metrics**
   - CPU/Memory utilization
   - Disk I/O
   - Network bandwidth

4. **Business Metrics**
   - Investigation queue size
   - Cost per transaction
   - Fraud caught vs. missed

### Alerting Setup

Configure PagerDuty/Slack integration:

```yaml
# AlertManager configuration
receivers:
  - name: "pagerduty-critical"
    pagerduty_configs:
      - service_key: YOUR_PAGERDUTY_KEY
        severity: critical

  - name: "slack-warnings"
    slack_configs:
      - api_url: YOUR_SLACK_WEBHOOK
        channel: "#fraud-detection-alerts"

route:
  group_by: ["alertname", "severity"]
  receiver: "slack-warnings"
  routes:
    - match:
        severity: critical
      receiver: "pagerduty-critical"
```

---

## Disaster Recovery

### Backup Strategy

#### 1. Database Backups

```bash
# Automated PostgreSQL backups
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \
  --restart=Never \
  -- /bin/bash -c "pg_dump -h postgres -U fraud_detection_user fraud_detection | gzip > /backup/fraud_detection_$(date +%Y%m%d).sql.gz"
```

#### 2. Model Backups

```bash
# Sync models to S3/GCS daily
aws s3 sync /app/models/ s3://fraud-detection-models/backup/$(date +%Y%m%d)/
```

### Disaster Recovery Plan

1. **Database Failure**

   ```bash
   # Restore from backup
   kubectl exec -it postgres-0 -- psql -U fraud_detection_user < backup.sql
   ```

2. **Complete Cluster Failure**
   - Provision new cluster
   - Restore from backups
   - Redirect DNS
   - Estimated RTO: 2-4 hours

3. **Model Corruption**
   - Rollback to previous version
   - Restore from S3/GCS backup
   - Estimated RTO: 15-30 minutes

---

## Security

### Network Security

1. **Enable Network Policies**

   ```bash
   kubectl apply -f k8s/ingress-policies.yaml
   ```

2. **TLS/SSL Configuration**
   - Use cert-manager for automatic certificate management
   - Enforce HTTPS-only traffic
   - Rotate certificates every 90 days

3. **API Authentication**
   - Implement API key authentication
   - Use OAuth 2.0 for user access
   - Rate limiting per API key

### Data Security

1. **PII Protection**
   - All PII redacted before LLM processing
   - Encrypted at rest (database encryption)
   - Encrypted in transit (TLS 1.3)

2. **Audit Logging**
   - Log all predictions
   - Log all data access
   - Retain logs for 7 years (compliance)

3. **Secrets Management**
   - Use Kubernetes Secrets or HashiCorp Vault
   - Rotate credentials every 90 days
   - Never commit secrets to git

---

## Performance Tuning

### 1. Model Optimization

```python
# Use model quantization for faster inference
from sklearn.tree import DecisionTreeClassifier
import joblib

# Save optimized model
joblib.dump(model, 'model_optimized.pkl', compress=3)
```

### 2. Caching Strategy

```python
# Redis caching for feature engineering
import redis
r = redis.Redis(host='redis', port=6379)

def get_user_features(user_id):
    cache_key = f"user:{user_id}:features"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    features = compute_features(user_id)
    r.setex(cache_key, 3600, json.dumps(features))
    return features
```

### 3. Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_predictions_fraud_score ON predictions(fraud_score);
```

### 4. Connection Pooling

```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)
```

---

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check resource limits
   - Review model complexity
   - Optimize database queries
   - Enable caching

2. **Out of Memory**
   - Increase memory limits
   - Check for memory leaks
   - Reduce batch size
   - Enable model compression

3. **Pods Crashing**
   - Check logs: `kubectl logs`
   - Review resource requests
   - Verify model loading
   - Check database connectivity

4. **Performance Degradation**
   - Check for concept drift
   - Review recent data changes
   - Trigger model retraining
   - Scale up resources

---

## Production Checklist

Before going live:

- [ ] All secrets properly configured
- [ ] TLS certificates issued
- [ ] Monitoring dashboards created
- [ ] Alerts configured and tested
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Security scan completed
- [ ] Load testing completed
- [ ] Documentation up to date
- [ ] Team trained on operations
- [ ] Runbook created
- [ ] On-call rotation established

---

## Support

For issues and questions:

- **Technical Support**: tech-support@your-domain.com
- **On-Call**: PagerDuty escalation
- **Documentation**: https://docs.your-domain.com
- **Slack**: #fraud-detection-support

---

_Last Updated: 2024_
_Version: 1.0.0_
