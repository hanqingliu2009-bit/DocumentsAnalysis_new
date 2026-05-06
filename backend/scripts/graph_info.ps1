param(
  [string]$Uri = "http://14.103.133.160:8022/graph/info",
  [string]$Query = "高速智能封边机维护保养手册适用于哪些产品型号？",
  [string]$Domain = "南兴装备",
  [ValidateSet("triplet", "chunk", "summary")]
  [string]$SearchType = "triplet",
  [hashtable]$Extra = @{}
)

$ErrorActionPreference = "Stop"

$payload = @{
  query       = $Query
  domain      = $Domain
  search_type = $SearchType
}

if ($Extra) {
  foreach ($k in $Extra.Keys) {
    $payload[$k] = $Extra[$k]
  }
}

$body = $payload | ConvertTo-Json -Depth 10 -Compress

try {
  Invoke-RestMethod -Method Post `
    -Uri $Uri `
    -ContentType "application/json" `
    -Headers @{ Accept = "application/json" } `
    -Body $body
} catch {
  Write-Host "Request payload:" -ForegroundColor DarkGray
  Write-Host $body -ForegroundColor DarkGray

  $ex = $_.Exception
  $resp = $ex.Response
  if ($null -ne $resp) {
    try {
      $statusCode = [int]$resp.StatusCode
      $statusDesc = $resp.StatusDescription
      $stream = $resp.GetResponseStream()
      $reader = New-Object System.IO.StreamReader($stream)
      $raw = $reader.ReadToEnd()
      Write-Host "HTTP $statusCode $statusDesc" -ForegroundColor Yellow
      if ($raw -and $raw.Trim()) {
        Write-Host $raw
      } elseif ($_.ErrorDetails -and $_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message
      } else {
        Write-Host "(empty response body)"
      }
      exit 1
    } catch {
      throw
    }
  }
  throw
}

