import os
import time
import subprocess
import signal
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests


# === ENHANCED CONFIG ===
ISTIO_NAMESPACE = "istio-system"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_SERVICE_NAME = "prometheus-kube-prometheus-prometheus"
PROMETHEUS_PORT = 9090
REQUIRED_ISTIO_PODS = ["istiod", "istio-ingressgateway"]
KIALI_SERVICE_NAME = "kiali"
KIALI_NAMESPACE = "istio-system"
KIALI_LOCAL_PORT = 20001
# Additional services to check and port-forward
SERVICES_TO_FORWARD = {
    "prometheus": {
        "namespace": "monitoring",
        "service": "prometheus-kube-prometheus-prometheus",
        "local_port": 9090,
        "target_port": 9090
    },
    "grafana": {
        "namespace": "monitoring", 
        "service": "prometheus-grafana",
        "local_port": 3000,
        "target_port": 80
    },
    "kiali": {
        "namespace": "istio-system",
        "service": "kiali",
        "local_port": 20001,
        "target_port": 20001
    },
    "jaeger": {
        "namespace": "istio-system",
        "service": "tracing",
        "local_port": 16686,
        "target_port": 80
    }
}

ONLINEBOUTIQUE_NAMESPACE = "default"  # or wherever you deploy it
REQUIRED_ONLINEBOUTIQUE_SERVICES = [
    "frontend", "cartservice", "productcatalogservice", 
    "currencyservice", "paymentservice", "shippingservice",
    "emailservice", "checkoutservice", "recommendationservice", "adservice"
]

METRIC_DESCRIPTIONS = {
    "up": "Target health (1=up)",
    "istio_requests_total": "Total Istio requests",
    "istio_request_duration_seconds_bucket": "Request latency (histogram)",
    "container_cpu_usage_seconds_total": "Container CPU usage",
    "container_memory_usage_bytes": "Container memory usage",
    "node_cpu_seconds_total": "Node CPU stats",
    "node_memory_MemAvailable_bytes": "Available memory on nodes",
    "prometheus_engine_query_duration_seconds_count": "Prometheus query counts"
}


def list_cluster_nodes():
    print("\n[📦] Cluster Nodes:")
    v1 = client.CoreV1Api()
    try:
        nodes = v1.list_node().items
        for node in nodes:
            name = node.metadata.name
            status = next((s.status for s in node.status.conditions if s.type == "Ready"), "Unknown")
            arch = node.status.node_info.architecture
            print(f"   - {name} | Status: {status} | Arch: {arch}")
    except Exception as e:
        print("[✗] Failed to list nodes:", e)

def list_all_pods():
    print("\n[📦] Cluster Pods:")
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_pod_for_all_namespaces().items
        for pod in pods:
            ns = pod.metadata.namespace
            name = pod.metadata.name
            status = pod.status.phase
            node = pod.spec.node_name
            print(f"   - [{ns}] {name} | Status: {status} | Node: {node}")
    except Exception as e:
        print("[✗] Failed to list pods:", e)


def load_kube_config():
    try:
        config.load_kube_config()
        print("[✓] Loaded local kubeconfig.")
    except:
        try:
            config.load_incluster_config()
            print("[✓] Loaded in-cluster kubeconfig.")
        except Exception as e:
            print("[✗] Failed to load kube config:", e)
            exit(1)

def check_istio_pods():
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace=ISTIO_NAMESPACE).items
        for required in REQUIRED_ISTIO_PODS:
            found = any(required in pod.metadata.name and pod.status.phase == "Running" for pod in pods)
            if found:
                        print(f"[✓] Istio pod '{required}' is running.")
            else:
                print(f"[✗] Istio pod '{required}' is NOT running.")
    except Exception as e:
        print(f"[✗] Error checking Istio pods: {e}")

def check_prometheus_service():
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(PROMETHEUS_SERVICE_NAME, PROMETHEUS_NAMESPACE)
        print(f"[✓] Found Prometheus service '{PROMETHEUS_SERVICE_NAME}' in namespace '{PROMETHEUS_NAMESPACE}'")
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Prometheus service '{PROMETHEUS_SERVICE_NAME}' not found in namespace '{PROMETHEUS_NAMESPACE}'")
        else:
            print(f"[✗] Error checking Prometheus service: {e}")

def start_port_forward(namespace, service_name, local_port, target_port, address="0.0.0.0"):
    print(f"[*] Starting port-forward: {address}:{local_port} -> {service_name}:{target_port}")
    cmd = [
        "kubectl", "port-forward",
        f"svc/{service_name}",
        f"{local_port}:{target_port}",
        "-n", namespace,
        "--address", address
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(2)  # allow it to start
    return proc


def check_kiali_service():
    """Check if Kiali service is available"""
    print("\n[🕸️] Checking Kiali service:")
    v1 = client.CoreV1Api()
    try:
        svc = v1.read_namespaced_service(KIALI_SERVICE_NAME, KIALI_NAMESPACE)
        print(f"[✓] Found Kiali service '{KIALI_SERVICE_NAME}' in namespace '{KIALI_NAMESPACE}'")
        
        # Also check if Kiali pod is running
        pods = v1.list_namespaced_pod(namespace=KIALI_NAMESPACE).items
        kiali_pod = next((pod for pod in pods if "kiali" in pod.metadata.name), None)
        if kiali_pod and kiali_pod.status.phase == "Running":
            print(f"[✓] Kiali pod is running")
        else:
            print(f"[⚠️] Kiali service found but pod may not be running")
        
        return True
    except ApiException as e:
        if e.status == 404:
            print(f"[✗] Kiali service '{KIALI_SERVICE_NAME}' not found in namespace '{KIALI_NAMESPACE}'")
            print("    💡 To install Kiali: kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml")
        else:
            print(f"[✗] Error checking Kiali service: {e}")
        return False

def start_kiali_port_forward():
    """Start port forwarding for Kiali"""
    print(f"\n[🚀] Starting Kiali port-forward...")
    try:
        proc = start_port_forward(
            KIALI_NAMESPACE,
            KIALI_SERVICE_NAME,
            KIALI_LOCAL_PORT,
            KIALI_TARGET_PORT
        )
        print(f"[✓] Kiali available at: http://localhost:{KIALI_LOCAL_PORT}")
        print("    📊 Use Kiali to visualize your service mesh topology and traffic")
        return proc
    except Exception as e:
        print(f"[✗] Failed to start Kiali port-forward: {e}")
        return None

def stop_port_forward(proc, service_name):
    """Stop a specific port-forward process"""
    try:
        if proc:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            print(f"[✓] Stopped {service_name} port-forward")
    except Exception as e:
        print(f"[✗] Error stopping {service_name} port-forward: {e}")



# ...existing code...

def main():
    """Main function to test Kiali functionality"""
    print("🔧 Kubernetes Cluster Checker - Kiali Test")
    print("=" * 50)
    
    # Load kubeconfig
    load_kube_config()
    
    # Check basic cluster info
    list_cluster_nodes()
    
    # Check Istio components
    print("\n[🔍] Checking Istio components:")
    check_istio_pods()
    
    # Check Kiali
    kiali_available = check_kiali_service()
    
    if kiali_available:
        # Ask user if they want to start Kiali port-forward
        response = input(f"\nStart Kiali port-forward on localhost:{KIALI_LOCAL_PORT}? (y/N): ")
        if response.lower() == 'y':
            kiali_proc = start_kiali_port_forward()
            
            if kiali_proc:
                try:
                    print(f"\n[ℹ️] Kiali is running at http://localhost:{KIALI_LOCAL_PORT}")
                    print("[ℹ️] Press Ctrl+C to stop port-forward...")
                    signal.pause()
                except KeyboardInterrupt:
                    print("\n[🛑] Stopping Kiali port-forward...")
                    stop_port_forward(kiali_proc, "Kiali")
                    print("[👋] Cleanup complete!")
    else:
        print("\n[ℹ️] Kiali is not available. Install it first to proceed.")

if __name__ == "__main__":
    main()