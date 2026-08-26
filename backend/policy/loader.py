from pathlib import Path
from typing import Any
import yaml

from backend.shared.config import settings

class PolicyConfigurationError(ValueError):
    """Raised when a policy file cannot be safely evaluated."""

class PolicyLoader:
    def __init__(self, policies_dir: str = settings.policies_dir):
        self.policies_dir = Path(policies_dir)
        self.policy_sets = []
        self.reload()

    def reload(self) -> None:
        if not self.policies_dir.is_dir():
            raise PolicyConfigurationError(f"Policies directory not found: {self.policies_dir}")
            
        loaded_sets = []
        allowed_actions = {"ALLOW", "MODIFY", "REROUTE", "HUMAN_REVIEW", "BLOCK"}
        
        for file_path in self.policies_dir.glob("*.yaml"):
            with file_path.open("r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
                
            # Skip specialized agent tool policies (loaded by backend.agents.policy)
            if policy.get("scope") == "agent_tools":
                continue

            if not isinstance(policy, dict) or not isinstance(policy.get("rules"), list):
                raise PolicyConfigurationError(f"Policy {file_path.name} must contain a top-level rules list.")
                
            for rule in policy["rules"]:
                if not isinstance(rule, dict) or not {"id", "when", "action"} <= rule.keys():
                    raise PolicyConfigurationError(f"Rule in {file_path.name} needs id, when, and action.")
                if not isinstance(rule["when"], dict):
                    raise PolicyConfigurationError(f"Rule {rule['id']} has a non-object when condition.")
                if rule["action"] not in allowed_actions:
                    raise PolicyConfigurationError(f"Rule {rule['id']} has an unsupported action.")
                if "priority" in rule:
                    try:
                        int(rule["priority"])
                    except (TypeError, ValueError) as exc:
                        raise PolicyConfigurationError(f"Rule {rule['id']} has an invalid priority.") from exc
            
            default_action = policy.get("defaults", {}).get("action", "ALLOW")
            if default_action not in allowed_actions:
                raise PolicyConfigurationError(f"The default policy action in {file_path.name} is unsupported.")
                
            loaded_sets.append({
                "policy_set": policy.get("name", file_path.stem),
                "scope": policy.get("scope", "global"),
                "rules": policy["rules"],
                "default_action": default_action,
                "version": str(policy.get("version", "unknown"))
            })
            
        self.policy_sets = loaded_sets
