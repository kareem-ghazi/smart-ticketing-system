[CmdletBinding()]
param(
    [switch]$Once,
    [int]$PollSeconds = 1
)

$ErrorActionPreference = "Stop"

$planDirectory = $PSScriptRoot
$masterPath = Join-Path $planDirectory "smart-ticketing-system.md"
$syncStart = "<!-- BEGIN MASTER PLAN SYNC -->"
$syncEnd = "<!-- END MASTER PLAN SYNC -->"

$componentMap = [ordered]@{
    "01-triage-ml-component.md" = @("Phase 1")
    "02-bilingual-llm-component.md" = @("Phase 2")
    "03-backend-component.md" = @("Phase 3")
    "04-frontend-and-deployment-component.md" = @("Phase 4", "Phase 5")
}

function Get-PlanSection {
    param(
        [string]$PlanText,
        [string]$Heading
    )

    $escapedHeading = [regex]::Escape($Heading)
    $match = [regex]::Match(
        $PlanText,
        "(?ms)^##\s+$escapedHeading\b.*?(?=^##\s+|\z)"
    )

    if (-not $match.Success) {
        throw "Could not find '$Heading' in $masterPath."
    }

    return $match.Value.Trim()
}

function Get-AlignmentContent {
    param(
        [string]$PlanText,
        [string[]]$Headings
    )

    $sections = foreach ($heading in $Headings) {
        Get-PlanSection -PlanText $PlanText -Heading $heading
    }

    return @(
        $syncStart
        "## Master plan alignment"
        ""
        "> This section is generated from [smart-ticketing-system.md](./smart-ticketing-system.md). Do not edit it directly."
        ""
        ($sections -join "`r`n`r`n")
        ""
        $syncEnd
    ) -join "`r`n"
}

function Update-Component {
    param(
        [string]$Path,
        [string]$AlignmentContent
    )

    $existing = if (Test-Path -LiteralPath $Path) {
        Get-Content -LiteralPath $Path -Raw
    } else {
        ""
    }

    $escapedStart = [regex]::Escape($syncStart)
    $escapedEnd = [regex]::Escape($syncEnd)
    $pattern = "(?ms)$escapedStart.*?$escapedEnd"

    if ([regex]::IsMatch($existing, $pattern)) {
        $updated = [regex]::Replace($existing, $pattern, $AlignmentContent, 1)
    } elseif ([string]::IsNullOrWhiteSpace($existing)) {
        $updated = $AlignmentContent + "`r`n"
    } else {
        $updated = $existing.TrimEnd() + "`r`n`r`n" + $AlignmentContent + "`r`n"
    }

    if ($updated -ne $existing) {
        [System.IO.File]::WriteAllText(
            $Path,
            $updated,
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Host "Updated $([System.IO.Path]::GetFileName($Path))"
    }
}

function Sync-Plan {
    $planText = Get-Content -LiteralPath $masterPath -Raw

    foreach ($entry in $componentMap.GetEnumerator()) {
        $componentPath = Join-Path $planDirectory $entry.Key
        $alignment = Get-AlignmentContent -PlanText $planText -Headings $entry.Value
        Update-Component -Path $componentPath -AlignmentContent $alignment
    }
}

if (-not (Test-Path -LiteralPath $masterPath)) {
    throw "Master plan not found: $masterPath"
}

Sync-Plan

if (-not $Once) {
    Write-Host "Watching $masterPath for changes. Press Ctrl+C to stop."
    $lastWrite = (Get-Item -LiteralPath $masterPath).LastWriteTimeUtc

    while ($true) {
        Start-Sleep -Seconds ([Math]::Max(1, $PollSeconds))
        $currentWrite = (Get-Item -LiteralPath $masterPath).LastWriteTimeUtc

        if ($currentWrite -ne $lastWrite) {
            Sync-Plan
            $lastWrite = $currentWrite
        }
    }
}
