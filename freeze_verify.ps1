$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\tools\freeze_verify.py" @args
