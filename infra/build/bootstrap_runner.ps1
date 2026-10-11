$ErrorActionPreference = 'Stop'
$ForwardedArguments = @($args)

if ($ForwardedArguments.Count -gt 0 -and $ForwardedArguments[0] -eq '--') {
    if ($ForwardedArguments.Count -eq 1) {
        $ForwardedArguments = @()
    }
    else {
        $ForwardedArguments = $ForwardedArguments[1..($ForwardedArguments.Count - 1)]
    }
}

if ($ForwardedArguments.Count -lt 1) {
    [Console]::Error.WriteLine('Bootstrap adapter requires an executable path followed by its arguments.')
    exit 2
}

$executable = $ForwardedArguments[0]
$executableArguments = @()
if ($ForwardedArguments.Count -gt 1) {
    $executableArguments = $ForwardedArguments[1..($ForwardedArguments.Count - 1)]
}

try {
    & $executable @executableArguments
    if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("Bootstrap executable failed with exit code ${LASTEXITCODE}: $executable")
        exit $LASTEXITCODE
    }
}
catch {
    [Console]::Error.WriteLine("Could not invoke bootstrap executable '$executable': $($_.Exception.Message)")
    exit 1
}

exit 0
