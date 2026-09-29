#!/bin/zsh
cd -- "${0:A:h}" || exit 1
/opt/anaconda3/envs/sc_dashboard2/bin/python final_results.py --run --sweep --k-min 3 --k-max 10
result=$?
echo ""
if [[ $result -ne 0 ]]; then
  echo "The sweep stopped. See the error above and the run.log in the results folder."
fi
read "?Press Return to close this window."
exit $result
