#Run with
#in anaconda prompt: conda init powershell
#powershell -ExecutionPolicy Bypass -File .\copy_radar.ps1

# ==============================
# SETTINGS
# ==============================

$pythonScript = "D:\scripts\process_old_radar_server.py"
$condaEnv = "radar"

$logDir = "D:\radar_logs"

#TO DO: change to locations of drives
$drives = @(
    "E:\",
    "F:\",
    "G:\"
    # "H:\",
    # "I:\",
    # "J:\"
)

#subdirectory structure underneath projDir
#TO DO -- change this to correct path
$subDir = "data\precip"

#TO DO: do all get saved into same out folder currently? If not need to make changes
$outDir = "D:\radar_data"

# Threads per drive (start with 2 for 6-7 USB HDDs)
$threads = 2

# ==============================
# COPY FUNCTION
# ==============================

$copyDrive = {
    param($drive, $outDir, $subDir, $threads, $logDir)
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logFile = Join-Path $logDir ($drive.TrimEnd('\').TrimEnd(':') + ".log")
    while ($true) {

        Write-Host "Searching $drive"

        # Find projDir anywhere on drive
        $project = Get-ChildItem $drive -Directory -Recurse `
            -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -eq "projDir" -and
                (Test-Path (Join-Path $_.FullName $subDir))
            } |
            Select-Object -First 1

        if (-not $project) {
            Write-Host "[$drive] Source not found. Retrying in 15 seconds..."
            Start-Sleep -Seconds 15
            continue
        }

        $dataDir = Join-Path $project.FullName $subDir
        Write-Host "Found: $dataDir"

        # Each subfolder is one day
        $failed = $false

        foreach ($day in (Get-ChildItem $dataDir -Directory)) {

            $source = $day.FullName
            $dest = Join-Path $outDir $day.Name

            Write-Host "Copying $($day.Name) from $drive"

            robocopy $source $dest *.mdv `
                /MT:$threads `
                /R:0 /W:0 `
                /NDL /NJH /NJS /NP `
                /LOG+:$logFile

            if ($LASTEXITCODE -ge 8) {
                Write-Warning "Some files failed in: $source"

                if (-not (Test-Path $drive)) {
                    $failed = $true
                    break
                }
            }
        }

        if (-not $failed -and (Test-Path $drive)) {
            Write-Host "[$drive] Finished"
            break
        }

        Write-Host "[$drive] Transfer interrupted. Retrying in 15 seconds..."
        Start-Sleep -Seconds 15
    }
}

# ==============================
# RUN ALL DRIVES IN PARALLEL
# ==============================

$jobs = foreach ($drive in $drives) {
    Start-Job -ScriptBlock $copyDrive `
        -ArgumentList $drive, $outDir, $subDir, $threads, $logDir
}

$jobs | Wait-Job | Receive-Job
$jobs | Remove-Job

Write-Host "All drives finished."

# Run Python processing
Write-Host "Starting radar processing..."

conda run --no-capture-output -n $condaEnv python -u $pythonScript

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Python processing failed."
} else {
    Write-Host "Radar processing completed."
}
