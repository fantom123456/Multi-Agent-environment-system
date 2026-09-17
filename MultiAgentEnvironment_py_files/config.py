"""
Central simulation configuration and reward parameters.
"""

# Task completion & contribution rewards
R_TASK_COMPLETE: float = 10.0
R_TASK_PROGRESS: float = 0.2
R_INVITE_SUCCESS: float = 2.5   # Increased from 1.0 to ensure collab is favored over passive replying

# Penalties and costs
C_SEND: float = 0.02
C_BROADCAST_EXTRA: float = 0.08
C_STEP: float = 0.001
P_FAIL: float = 5.0

# LinUCB exploration parameter
LINUCB_ALPHA: float = 2.5        # Increased from 1.5 to prevent premature convergence to 0 sent messages
LINUCB_LAMBDA: float = 1.0
LINUCB_DIM: int = 8