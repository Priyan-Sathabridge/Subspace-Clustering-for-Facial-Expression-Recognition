#!/bin/zsh
cd -- "${0:A:h}" || exit 1
/opt/anaconda3/envs/sc_dashboard2/bin/python final_results.py --run
result=$?
echo ""
if [[ $result -ne 0 ]]; then
  echo "The run stopped. See the error above and the run.log in the results folder."
fi
read "?Press Return to close this window."
exit $result
