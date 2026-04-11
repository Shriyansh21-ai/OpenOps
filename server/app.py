"""OpenOps: Server Entry Point"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openenv.core.env_server import create_app
from models import SupportAction, SupportObservation
from environment import OpenOpsEnvironment

app = create_app(
    env=OpenOpsEnvironment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="openops",
)

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
