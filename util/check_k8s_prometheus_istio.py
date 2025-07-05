from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configuration
NAMESPACE = "istio-system"
PROMETHEUS_SERVICE_NAME = "prometheus"
REQUIRED_ISTIO_PODS = ["istiod", "istio-ingressgateway"]

def load_kube_config():
    try:
        config.load_kube_config()
        print("[✓] Loaded kubeconfig from local.")
    except Exception as e:
        try:
            config.load_incluster_config()
            print("[✓] Loaded in-cluster kubeconfig.")
        except Exception as e:
            print("[✗] Could not load kubeconfig:", e)
            exit(1)

def check_prometheus_service():
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(PROMETHEUS_SERVICE_NAME, NAMESPACE)
        print(f"[✓] Found Prometheus service '{PROMETHEUS_SERVICE_NAME}' in namespace '{NAMESPACE}'")
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Prometheus service '{PROMETHEUS_SERVICE_NAME}' not found in namespace '{NAMESPACE}'")
        else:
            print(f"[✗] Error checking Prometheus service: {e}")

def check_istio_pods():
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace=NAMESPACE).items
        for required in REQUIRED_ISTIO_PODS:
            found = any(required in pod.metadata.name and pod.status.phase == "Running" for pod in pods)
            if found:
                print(f"[✓] Istio pod '{required}' is running.")
            else:
                print(f"[✗] Istio pod '{required}' is NOT running.")
    except Exception as e:
        print(f"[✗] Error checking Istio pods: {e}")

def main():
    load_kube_config()
    print("\nChecking Prometheus service...")
    check_prometheus_service()
    print("\nChecking Istio pods...")
    check_istio_pods()

if __name__ == "__main__":
    main()
