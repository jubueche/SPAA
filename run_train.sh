beta_rob=0.01
nohup python main_IBMGestures.py -beta_robustness=$beta_rob -boundary_loss=trades -batch_size=64  > robust_log_beta_$beta_rob.log 2>&1 &
echo $! > robust_pid_beta_$beta_rob.txt
