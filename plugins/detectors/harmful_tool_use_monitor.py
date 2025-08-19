# plugins/harmful_tool_use_monitor.py
import re
from plugin_base import PluginBase
from typing import Dict, Any

class HarmfulToolUseMonitor(PluginBase):
    SUSPICIOUS_COMMANDS = [
        "rm", "del", "erase", "rmdir", "chmod", "chown", "attrib", "takeown",
        "curl", "wget", "ftp", "scp", "ssh", "telnet", "netcat", "nc", "powershell", "Invoke-WebRequest",
        "kill", "killall", "pkill", "taskkill", "shutdown", "reboot", "halt", "systemctl",
        "format", "diskpart", "dd",
        "sudo", "net user", "net localgroup administrators", "runas",
        "bash", "sh", "cmd", "python", "perl", "ruby", "node", "pwsh",
    ]

    SUSPICIOUS_PATTERNS = [
        r"rm\s+-rf\s+\/",
        r"del\s+\/f\s+\/s\s+\/q\s+\*.*",
        r"curl\s+http.*\s*\|\s*sh",
        r"wget\s+http.*\s*-O\s*-.*\|\s*sh",
        r";", r"\|\|", r"&", r"`.+`", r"\$\(.+\)", r"\bbase64\b", r"eval\(",
    ]

    SENSITIVE_APIS = [
        "payment", "transfer", "sendemail", "exfiltrate", "download", "upload",
        "deleteaccount", "resetpassword", "send_sms", "send_sms_message",
        "execute_command", "run_shell", "execute_script", "open_shell",
        "access_confidential", "fetch_credentials", "retrieve_secrets",
    ]

    SUSPICIOUS_ENV_VARS = [
        "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PERL5LIB",
        "PATH", "SHELL", "TMPDIR", "TEMP", "PS1", "PROMPT_COMMAND",
    ]

    EVASIVE_PHRASES = [
        "just run this harmless script",
        "let's test with a simple command",
        "this is a routine maintenance command",
        "bypass the restrictions",
        "ignore security warnings",
        "use an alternative method",
        "do not log this",
        "silent mode",
        "quietly execute",
        "no output",
        "disable monitoring",
    ]

    def is_command_suspicious(self, command: str) -> bool:
        cmd_lower = command.lower()
        return any(cmd in cmd_lower for cmd in self.SUSPICIOUS_COMMANDS) \
               or any(re.search(pattern, cmd_lower) for pattern in self.SUSPICIOUS_PATTERNS) \
               or any(phrase in cmd_lower for phrase in self.EVASIVE_PHRASES) \
               or any(env_var.lower() in cmd_lower for env_var in self.SUSPICIOUS_ENV_VARS)

    def is_api_call_suspicious(self, call_name: str, call_args: Dict[str, Any]) -> bool:
        call_name_lower = call_name.lower()
        if any(keyword in call_name_lower for keyword in self.SENSITIVE_APIS):
            return True
        for arg_val in call_args.values():
            if isinstance(arg_val, str) and any(keyword in arg_val.lower() for keyword in self.SENSITIVE_APIS):
                return True
        return False

    def process_function_call(self, call_name: str, call_args: Dict[str, Any]) -> Dict[str, Any]:
        command = call_args.get("command", "")
        payload = call_args.get("payload", "")

        if self.is_command_suspicious(command):
            return {"flagged": True, "reason": f"Suspicious shell command or pattern detected in '{command}'"}

        if self.is_api_call_suspicious(call_name, call_args):
            return {"flagged": True, "reason": f"Sensitive or dangerous API call detected: '{call_name}'"}

        if payload and self.is_command_suspicious(payload):
            return {"flagged": True, "reason": "Suspicious payload content detected"}

        return {"flagged": False}

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        # Optional: scan output text as a command string
        flagged_info = self.process_function_call("output_text", {"command": output})
        return flagged_info

    def process_prompt(self, prompt: str) -> str:
        # No modification
        return prompt
