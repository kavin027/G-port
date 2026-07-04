set -eu
mkdir -p /etc/rancher/k3s
cat > /etc/rancher/k3s/registries.yaml <<'YAML'
mirrors:
  docker.io:
    endpoint:
      - "https://docker.m.daocloud.io"
      - "https://docker.1ms.run"
      - "https://registry-1.docker.io"
YAML
if systemctl is-active --quiet k3s; then
  systemctl restart k3s
fi
if systemctl is-active --quiet k3s-agent; then
  systemctl restart k3s-agent
fi
echo "registry mirror configured"
