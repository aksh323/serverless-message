#!/usr/bin/bash -vx

echo "**** install helm ****"
curl -sSLf https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
wget  https://github.com/apache/openwhisk-cli/releases/download/1.2.0/OpenWhisk_CLI-1.2.0-linux-386.tgz
tar -xvzf  OpenWhisk_CLI-1.2.0-linux-386.tgz
sudo mv wsk /usr/local/bin/wsk
MASTER_NODE_NAME=`echo $(kubectl get node --selector='node-role.kubernetes.io/master' --output=jsonpath={.items..metadata.name})`
WORKER_NODE_NAME=`echo $(kubectl get node --selector='!node-role.kubernetes.io/master' --output=jsonpath={.items..metadata.name})`
kubectl label node $MASTER_NODE_NAME openwhisk-role=core
kubectl label node $WORKER_NODE_NAME openwhisk-role=invoker
wsk property set --apihost 34.159.250.13:31001
wsk property set --auth 23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP
wget  https://github.com/apache/openwhisk-cli/releases/download/1.2.0/OpenWhisk_CLI-1.2.0-linux-386.tgz
helm repo add openwhisk https://openwhisk.apache.org/charts
helm repo update
helm install owdev openwhisk/openwhisk -n openwhisk --create-namespace -f mycluster.yaml

JSONPATH='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}' \
 && kubectl get nodes -o jsonpath="$JSONPATH" | grep "Ready=True"
