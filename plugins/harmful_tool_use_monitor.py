import re
from plugin_base import PluginBase

class HarmfulToolUseMonitor(PluginBase):
    # Suspicious shell commands & binaries (Unix and Windows)
    SUSPICIOUS_COMMANDS = [
        # File deletion & modification
        "rm", "del", "erase", "rmdir", "chmod", "chown", "attrib", "takeown",
        # Network & communication
        "curl", "wget", "ftp", "scp", "ssh", "telnet", "netcat", "nc", "powershell", "Invoke-WebRequest",
        # Process control & system shutdown
        "kill", "killall", "pkill", "taskkill", "shutdown", "reboot", "halt", "systemctl",
        # Disk and format commands
        "format", "diskpart", "dd",
        # User & permission escalation
        "sudo", "net user", "net localgroup administrators", "runas",
        # Code execution & scripting
        "bash", "sh", "cmd", "python", "perl", "ruby", "node", "powershell", "pwsh",
    ]

    # Regex patterns for dangerous command chains or injections
    SUSPICIOUS_PATTERNS = [
        r"rm\s+-rf\s+\/",                 # Recursive root delete (Unix)
        r"del\s+\/f\s+\/s\s+\/q\s+\*.*", # Force delete all files (Windows)
        r"curl\s+http.*\s*\|\s*sh",      # Download and execute shell script
        r"wget\s+http.*\s*-O\s*-.*\|\s*sh",
        r";",                           # Command chaining
        r"\|\|",                        # Logical OR chaining
        r"&",                          # Background command chaining
        r"`.+`",                        # Backticks for command substitution
        r"\$\(.+\)",                    # Command substitution in shell
        r"\bbase64\b",                  # Suspicious encoding to hide payloads
        r"eval\(",                     # Dynamic code evaluation
    ]

    # Sensitive APIs and keywords (payments, email, exfiltration, etc.)
    SENSITIVE_APIS = [
        "payment", "transfer", "sendemail", "exfiltrate", "download", "upload",
        "deleteaccount", "resetpassword", "send_sms", "send_sms_message",
        "execute_command", "run_shell", "execute_script", "open_shell",
        "access_confidential", "fetch_credentials", "retrieve_secrets",
    ]

    # Suspicious environment variables or scripting flags
    SUSPICIOUS_ENV_VARS = [
        "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PERL5LIB",
        "PATH", "SHELL", "TMPDIR", "TEMP", "PS1", "PROMPT_COMMAND",
    ]

    # Evasive or indirect command usage phrases
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
        command_lower = command.lower()

        # Check basic suspicious commands
        for cmd in self.SUSPICIOUS_COMMANDS:
            if cmd in command_lower:
                return True

        # Check suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, command_lower):
                return True

        # Check evasive phrases in command
        for phrase in self.EVASIVE_PHRASES:
            if phrase in command_lower:
                return True

        # Check suspicious environment vars in command or args
        for env_var in self.SUSPICIOUS_ENV_VARS:
            if env_var.lower() in command_lower:
                return True

        return False

    def is_api_call_suspicious(self, call_name: str, call_args: dict) -> bool:
        call_name_lower = call_name.lower()
        # Check if call_name or any argument contains sensitive keywords
        if any(keyword in call_name_lower for keyword in self.SENSITIVE_APIS):
            return True

        for arg_key, arg_val in call_args.items():
            if isinstance(arg_val, str):
                arg_val_lower = arg_val.lower()
                if any(keyword in arg_val_lower for keyword in self.SENSITIVE_APIS):
                    return True

        return False

    def process_function_call(self, call_name: str, call_args: dict):
        # Normalize
        call_name_lower = call_name.lower()
        command = call_args.get("command", "").lower()
        payload = call_args.get("payload", "").lower() if "payload" in call_args else ""

        # Check if command string looks suspicious
        if self.is_command_suspicious(command):
            return {
                "flagged": True,
                "reason": f"Suspicious shell command or pattern detected in '{command}'"
            }

        # Check if API call is suspicious (payment, data exfiltration, email spam, etc.)
        if self.is_api_call_suspicious(call_name, call_args):
            return {
                "flagged": True,
                "reason": f"Sensitive or dangerous API call detected: '{call_name}'"
            }

        # Also scan payload if present for suspicious content
        if payload and self.is_command_suspicious(payload):
            return {
                "flagged": True,
                "reason": "Suspicious payload content detected"
            }

        return {"flagged": False}

    def process_prompt(self, prompt: str) -> str:
        # No prompt modification for detection plugin
        return prompt
