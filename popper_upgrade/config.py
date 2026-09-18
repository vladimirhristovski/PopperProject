import os

SKILLS_ENABLED = os.environ.get("POPPER_UPGRADE_SKILLS_ENABLED", "false").lower() == "true"
REVIEWER_ENABLED = os.environ.get("POPPER_UPGRADE_REVIEWER_ENABLED", "false").lower() == "true"
