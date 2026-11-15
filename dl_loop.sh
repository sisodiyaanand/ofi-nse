cd ~/ofi-nse
for r in $(seq 1 30); do
  n=$(ls data/raw | grep -c -E 'SOL|BNB|XRP|DOGE')
  [ "$n" -ge 184 ] && break
  bash dl_v3.sh
  sleep 60
done
echo LOOPDONE
