ps -ef | grep lr_uci_dist | awk '{ print $2 }' | xargs kill -9 
ps -ef | grep network_dist | awk '{ print $2 }' | xargs kill -9 

