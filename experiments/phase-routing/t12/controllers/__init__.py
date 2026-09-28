"""T12 controller stubs (T1 + T2/T3 deepen)."""
from .act_stub import ACTStub, dry_call as act_dry_call
from .calibration import LOCK as CALIBRATION_LOCK, within_c_bar
from .convergence_stub import ConvergenceThresholdStub, dry_call as convergence_dry_call
from .coordinator_stub import CoordinatorRegimeStub, dry_call as coordinator_dry_call
from .coordinator_traj_stub import CoordinatorRegimeTrajStub, dry_call as coordinator_traj_dry_call
from .oracle_stub import OracleStub, dry_call as oracle_dry_call
from .fixed_stub import FixedStub, dry_call as fixed_dry_call
from .silencing import SilencingGate, apply_silence_to_decide_kwargs
from .stateless_regime_stub import StatelessRegimeStub, dry_call as stateless_dry_call

__all__ = [
    "ACTStub",
    "ConvergenceThresholdStub",
    "CoordinatorRegimeStub",
    "CoordinatorRegimeTrajStub",
    "OracleStub",
    "FixedStub",
    "StatelessRegimeStub",
    "SilencingGate",
    "CALIBRATION_LOCK",
    "within_c_bar",
    "act_dry_call",
    "convergence_dry_call",
    "coordinator_dry_call",
    "coordinator_traj_dry_call",
    "oracle_dry_call",
    "fixed_dry_call",
    "stateless_dry_call",
    "apply_silence_to_decide_kwargs",
]
