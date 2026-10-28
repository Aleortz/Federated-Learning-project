<#
 Lanzador (Windows/PowerShell). Siempre invoca los modulos Python del proyecto.
   .\scripts\fl.ps1 setup | data | partitions | server <iid|noniid> | client <ID> <IP> | test
                    | iid | noniid | disconnect | compare | all
 Extras tras el comando se pasan al modulo, p. ej.:  .\scripts\fl.ps1 server noniid --alpha 0.3
#>
param([Parameter(Position=0)][string]$Cmd = "help", [Parameter(ValueFromRemainingArguments=$true)]$Rest)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)     # phase2_hardware_semifl/, desde cualquier directorio
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
function Run { & $py @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
switch ($Cmd) {
  "setup"      { python -m venv .venv; & .venv\Scripts\python.exe -m pip install -r requirements.txt; exit $LASTEXITCODE }
  "data"       { Run -m scripts.preparar_datos @Rest }
  "partitions" { Run -m scripts.preparar_datos --both @Rest }
  "server"     { $s = $Rest[0]; Run -m master_edge.run_server --scheme $s --run-name $(if ($s -eq "noniid") {"non_iid"} else {$s}) @($Rest | Select-Object -Skip 1) }
  "client"     { Run -m worker_node.run_worker --id $Rest[0] --master $Rest[1] @($Rest | Select-Object -Skip 2) }
  "test"       { Run -m pytest -q @Rest }
  "iid"        { Run -m scripts.run_local --scheme iid --run-name iid @Rest }
  "noniid"     { Run -m scripts.run_local --scheme noniid --run-name non_iid @Rest }
  "disconnect" { Run -m scripts.run_resiliencia @Rest }
  "compare"    { Run -m scripts.comparar @Rest }
  "all"        { Run -m pytest -q; Run -m scripts.run_local --scheme iid --run-name iid; Run -m scripts.run_local --scheme noniid --run-name non_iid; Run -m scripts.run_resiliencia; Run -m scripts.comparar }
  default      { Get-Content $PSCommandPath -TotalCount 6 }
}
