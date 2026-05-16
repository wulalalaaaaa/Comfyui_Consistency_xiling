param(
    [string]$ProjectRoot = "E:\Project_python\Comfyui_Consistency_xiling",
    [string]$ComfyNodeRoot = "E:\draw\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_Consistency_xiling"
)

$src = Join-Path $ProjectRoot "_deps\sdscripts"
$dst = Join-Path $ComfyNodeRoot "_deps\sdscripts"

if (!(Test-Path $src)) {
    Write-Error "Source deps not found: $src"
    exit 1
}

New-Item -ItemType Directory -Force -Path $dst | Out-Null
robocopy $src $dst /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    Write-Error "robocopy failed with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output "Synced deps: $src -> $dst"
