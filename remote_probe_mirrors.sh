set -eu
for url in \
  https://docker.m.daocloud.io/v2/ \
  https://docker.1ms.run/v2/ \
  https://dockerproxy.com/v2/ \
  https://registry-1.docker.io/v2/
do
  printf "%s " "$url"
  curl -I --connect-timeout 5 --max-time 8 -s "$url" | head -1 || true
done
