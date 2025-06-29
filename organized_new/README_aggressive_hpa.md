# Aggressive HPA Testing for gym-hpa RL Training Data

This setup provides everything needed to generate rich training data for the gym-hpa reinforcement learning environment through aggressive autoscaling in a local Minikube deployment.

## 🎯 Objective
Create frequent scaling events (both up and down) to collect diverse state-action-reward sequences for RL training.

## 📁 Files Overview

### Main Scripts
- **`complete_aggressive_hpa_setup.sh`** - Complete automated setup and test runner
- **`aggressive-hpa.yaml`** - HPA configuration with ultra-low thresholds and fast scaling
- **`quick_aggressive_test.sh`** - 5-minute rapid load test with multiple phases
- **`run_advanced_load_test.sh`** - 20-minute comprehensive test with variable patterns
- **`advanced_variable_load.py`** - Advanced Locust script with realistic user behavior

### Supporting Scripts
- **`collect_metrics.sh`** - Continuous metrics collection during tests
- **`basic_metrics.sh`** - Basic system metrics collection
- **`enhanced_locust.py`** - Enhanced load testing script

## 🚀 Quick Start

### 1. Prerequisites Check
```bash
# Ensure Minikube is running
minikube status

# Ensure Online Boutique is deployed
kubectl get svc frontend-external

# Install Locust if needed
pip install locust
```

### 2. Run Complete Setup
```bash
cd /home/mohamed/Desktop/Pfe_new/organized_new
./complete_aggressive_hpa_setup.sh
```

This script will:
- ✅ Verify all prerequisites
- 💾 Backup existing HPA configurations
- ⚡ Apply aggressive HPA settings
- 🚀 Run your chosen load test
- 📊 Collect and display results
- 🔄 Offer restoration options

### 3. Alternative: Quick Test Only
```bash
# For a rapid 5-minute test
./quick_aggressive_test.sh

# For advanced 20-minute testing
./run_advanced_load_test.sh
```

## ⚡ Aggressive HPA Configuration

The aggressive HPA uses:
- **Ultra-low CPU thresholds**: 20-30% (vs default 50-80%)
- **Ultra-low memory thresholds**: 30-40% (vs default 70-80%)
- **Fast scale-up**: 15-25 seconds stabilization
- **Fast scale-down**: 30-60 seconds stabilization
- **High scaling rates**: Up to 100% increase per cycle

### Target Services
- `frontend` (1-10 replicas)
- `recommendationservice` (1-8 replicas)
- `productcatalogservice` (1-6 replicas)

## 🧪 Load Test Patterns

### Quick Test (5 minutes)
1. **Spike** to 80 users (30s)
2. **Drop** to 5 users (30s)
3. **Medium** load 40 users (45s)
4. **Second spike** to 100 users (30s)
5. **Gradual decrease** 60→40→20→10 (60s)

### Advanced Test (20 minutes)
1. **Variable Load Shapes** - Custom load curves (10 min)
2. **Oscillating Load** - Sine wave pattern (8 min)
3. **Random Spikes** - Unpredictable load bursts (10 min)

## 📊 Metrics Collection

During tests, the following data is automatically collected:

### HPA Metrics
- Current replica counts
- CPU/Memory utilization
- Scaling decisions and timing
- Target vs actual metrics

### Pod Metrics
- Resource usage per pod
- Pod creation/deletion events
- Startup and termination times

### Load Metrics
- Request patterns and response times
- Error rates and latencies
- User simulation behavior

### Files Generated
```
metrics_data/
├── quick_test_TIMESTAMP.log
├── advanced_test_TIMESTAMP/
├── backups/original_hpa_TIMESTAMP.yaml
└── locust_reports/
```

## 🔄 Restoration & Cleanup

The setup preserves original configurations:

```bash
# Restore original HPA
kubectl apply -f metrics_data/backups/original_hpa_*.yaml

# Remove all HPA
kubectl delete hpa --all

# View current state
kubectl get hpa
kubectl get pods
```

## 🎯 Expected Scaling Behavior

With aggressive settings, you should see:

### Scale-Up Events
- **Triggers**: CPU >20% or Memory >30%
- **Speed**: 15-25 seconds
- **Rate**: Double replicas per cycle

### Scale-Down Events  
- **Triggers**: CPU <20% and Memory <30% sustained
- **Speed**: 30-60 seconds
- **Rate**: 25-50% reduction per cycle

### Typical Sequence
```
1 pod → Load spike → 2 pods → More load → 4 pods → Load drop → 2 pods → 1 pod
```

## 📈 Monitoring Commands

### Real-time Monitoring
```bash
# Watch HPA status
watch -n 2 'kubectl get hpa'

# Monitor pod counts
watch -n 2 'kubectl get pods | grep -E "(frontend|recommendation|product)"'

# View resource usage
watch -n 5 'kubectl top pods'

# Check scaling events
kubectl get events --sort-by=.metadata.creationTimestamp | grep HPA
```

### Analysis Commands
```bash
# Detailed HPA information
kubectl describe hpa

# Recent scaling events
kubectl get events --field-selector type=Normal | grep -E "(Scaled|HorizontalPodAutoscaler)"

# Pod resource requests/limits
kubectl describe pods | grep -A 3 "Requests:"
```

## 🧠 RL Training Data Usage

The collected data provides:

### State Features
- Current CPU/Memory utilization
- Current replica count
- Load pattern characteristics
- Historical scaling decisions

### Actions
- Scale up (increase replicas)
- Scale down (decrease replicas)  
- No action (maintain current state)

### Rewards
- Positive: Appropriate scaling response
- Negative: Over/under scaling, SLA violations
- Cost considerations: Resource efficiency

### Data Format
Extract sequences like:
```
State(t) → Action(t) → State(t+1) → Reward(t+1)
```

## 🔧 Troubleshooting

### Common Issues

**HPA not scaling:**
```bash
# Check metrics server
kubectl top nodes
kubectl get apiservice v1beta1.metrics.k8s.io

# Verify resource requests are set
kubectl describe pods | grep -A 2 "Requests:"
```

**Frontend not accessible:**
```bash
# Get correct URL
minikube service frontend-external --url

# Check service
kubectl get svc frontend-external
```

**Load test failing:**
```bash
# Verify Locust installation
pip install locust

# Check frontend response
curl http://$(minikube ip):$(kubectl get svc frontend-external -o jsonpath='{.spec.ports[0].nodePort}')
```

## 🎓 Next Steps for RL Training

1. **Data Preprocessing**: Clean and format collected metrics
2. **Feature Engineering**: Extract relevant state features
3. **Reward Design**: Define appropriate reward functions
4. **Environment Setup**: Configure gym-hpa with your data
5. **Agent Training**: Train RL agents on collected experiences
6. **Validation**: Test trained agents on new scenarios

## 📚 References

- [gym-hpa GitHub](../gym-hpa-main/)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Online Boutique Demo](https://github.com/GoogleCloudPlatform/microservices-demo)
- [Locust Documentation](https://locust.io/)

---
**Ready to generate aggressive autoscaling data for your RL training!** 🚀
